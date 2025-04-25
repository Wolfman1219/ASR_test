# server.py
import numpy as np
import torch
import grpc
from concurrent import futures

# Import the generated protobuf code
import vad_pb2
import vad_pb2_grpc

# Configure Silero VAD
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                             model='silero_vad')

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

# Audio parameters
SAMPLE_RATE = 16000
# Silero VAD requires exactly 512 samples per chunk at 16kHz
VAD_CHUNK_SIZE = 512

class VADProcessor:
    def __init__(self):
        self.vad_iterator = VADIterator(model, threshold=0.5, sampling_rate=SAMPLE_RATE)
        self.speech_detected = False
        self.accumulated_audio = []
        
    def process_chunk(self, audio_chunk):
        try:
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            
            # Check if we have the right number of samples
            if len(audio_int16) != VAD_CHUNK_SIZE:
                print(f"Warning: Expected {VAD_CHUNK_SIZE} samples, got {len(audio_int16)}")
                # Pad or truncate to the right size
                if len(audio_int16) > VAD_CHUNK_SIZE:
                    audio_int16 = audio_int16[:VAD_CHUNK_SIZE]
                else:
                    padding = np.zeros(VAD_CHUNK_SIZE - len(audio_int16), dtype=np.int16)
                    audio_int16 = np.concatenate([audio_int16, padding])
            
            # Convert to float32 and normalize
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Process through VAD
            # Make sure we have a properly shaped tensor for VAD input
            audio_tensor = torch.from_numpy(audio_float32)
            
            # The VAD model expects a specific format
            if audio_tensor.dim() == 1:
                # Ensure it's the right shape (a single batch)
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Apply the VAD model directly instead of using the iterator
            # This is more reliable in some cases
            speech_prob = model(audio_tensor, SAMPLE_RATE).item()
            
            # Record audio for potential STT processing
            self.accumulated_audio.append(audio_chunk)
            
            # Use probability threshold to determine speech
            speech_detected_now = speech_prob > 0.5
            
            # Check for speech events
            if speech_detected_now and not self.speech_detected:
                self.speech_detected = True
                return "start"
            elif not speech_detected_now and self.speech_detected:
                self.speech_detected = False
                return "end"
            else:
                return "continue"
                
        except Exception as e:
            print(f"Error in process_chunk: {e}")
            # Return continue to avoid breaking the stream
            return "continue"
    
    def get_accumulated_audio(self):
        """Returns the accumulated audio and clears the buffer"""
        combined_audio = b''.join(self.accumulated_audio)
        self.accumulated_audio = []
        return combined_audio
    
    def reset(self):
        """Reset the VAD state"""
        if hasattr(self, 'vad_iterator'):
            self.vad_iterator.reset_states()
        self.speech_detected = False
        self.accumulated_audio = []


class VADServicer(vad_pb2_grpc.VADServiceServicer):
    def __init__(self):
        self.client_processors = {}
        
    def ProcessAudio(self, request_iterator, context):
        """Process streaming audio and return VAD events"""
        # Extract client ID from metadata or create a new one
        client_id = context.peer()
        
        # Create a processor for this client if it doesn't exist
        if client_id not in self.client_processors:
            self.client_processors[client_id] = VADProcessor()
        
        processor = self.client_processors[client_id]
        
        try:
            for request in request_iterator:
                # Process the audio chunk
                audio_chunk = request.audio_data
                status = processor.process_chunk(audio_chunk)
                
                # If speech ended, we can send the accumulated audio for STT processing
                if status == "end":
                    audio_data = processor.get_accumulated_audio()
                    
                    # Here you would typically send the audio to your STT service
                    # For demo purposes, we'll just acknowledge it
                    yield vad_pb2.VADResponse(
                        event=status,
                        message=f"Speech ended, {len(audio_data)} bytes processed for STT"
                    )
                elif status == "start":
                    yield vad_pb2.VADResponse(
                        event=status,
                        message="Speech detected"
                    )
                # We could yield continue events too, but that would create a lot of network traffic
                # Only uncomment if you need continuous feedback
                # elif status == "continue":
                #     yield vad_pb2.VADResponse(
                #         event=status,
                #         message="Processing..."
                #     )
        except Exception as e:
            print(f"Error processing audio for client {client_id}: {e}")
            context.abort(grpc.StatusCode.INTERNAL, f"Error: {str(e)}")
        finally:
            # Clean up if the client disconnects
            if client_id in self.client_processors:
                self.client_processors[client_id].reset()
    
    def ResetVAD(self, request, context):
        """Reset the VAD state for a client"""
        client_id = context.peer()
        if client_id in self.client_processors:
            self.client_processors[client_id].reset()
        return vad_pb2.ResetResponse(success=True)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vad_pb2_grpc.add_VADServiceServicer_to_server(VADServicer(), server)
    server.add_insecure_port('[::]:50055')
    server.start()
    print("Server started on port 50055")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped")


if __name__ == '__main__':
    serve()
