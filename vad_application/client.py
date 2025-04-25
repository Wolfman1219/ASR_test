# file_client.py
import io
import time
import wave
import grpc
import threading
import queue
import argparse
import numpy as np
import os

# Import the generated protobuf code
import vad_pb2
import vad_pb2_grpc

# Audio parameters
SAMPLE_RATE = 16000
# Silero VAD requires exactly 512 samples per chunk at 16kHz
VAD_CHUNK_SIZE = 512

class FileVADClient:
    def __init__(self, server_address='localhost:50051', wav_file=None):
        self.server_address = server_address
        self.wav_file = wav_file
        self.running = False
        self.vad_events_queue = queue.Queue()
        self.stt_queue = queue.Queue()
    
    def read_wav_file(self):
        """Generator that yields chunks from a WAV file with correct size for Silero VAD"""
        try:
            with wave.open(self.wav_file, 'rb') as wf:
                # Validate audio format
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                    print(f"Warning: WAV file should be mono, 16-bit, {SAMPLE_RATE}Hz")
                    print(f"Current file: channels={wf.getnchannels()}, sampwidth={wf.getsampwidth()}, framerate={wf.getframerate()}")
                
                # Read chunks of exactly VAD_CHUNK_SIZE samples (each sample is 2 bytes for 16-bit audio)
                bytes_per_chunk = VAD_CHUNK_SIZE * wf.getsampwidth()
                
                print(f"Processing audio in chunks of {VAD_CHUNK_SIZE} samples ({bytes_per_chunk} bytes)")
                
                # Read the first chunk
                data = wf.readframes(VAD_CHUNK_SIZE)
                
                while data and self.running:
                    # Check if we have a complete chunk
                    if len(data) == bytes_per_chunk:
                        yield data
                    elif len(data) > 0:
                        # If we have a partial chunk, pad it with zeros
                        padding_needed = bytes_per_chunk - len(data)
                        padded_data = data + b'\x00' * padding_needed
                        print(f"Padded final chunk from {len(data)} to {bytes_per_chunk} bytes")
                        yield padded_data
                    
                    # Add a small delay to simulate real-time processing
                    time.sleep(VAD_CHUNK_SIZE / SAMPLE_RATE)  # Sleep for the duration of the chunk
                    
                    # Read the next chunk
                    data = wf.readframes(VAD_CHUNK_SIZE)
                
                print("Finished reading file")
                
        except Exception as e:
            print(f"Error reading WAV file: {e}")

    def stream_file(self):
        """Stream audio from WAV file to VAD service"""
        self.running = True
        
        # Verify file exists
        if not os.path.exists(self.wav_file):
            print(f"Error: File '{self.wav_file}' not found")
            return
            
        # Create a gRPC channel
        with grpc.insecure_channel(self.server_address) as channel:
            stub = vad_pb2_grpc.VADServiceStub(channel)
            
            # Function to generate audio chunks for streaming
            def generate_audio_chunks():
                for audio_chunk in self.read_wav_file():
                    yield vad_pb2.AudioChunk(audio_data=audio_chunk)
            
            # Process responses from server
            try:
                for response in stub.ProcessAudio(generate_audio_chunks()):
                    event = response.event
                    message = response.message
                    
                    # Put event in queue for processing by main application
                    self.vad_events_queue.put((event, message))
                    
                    # Print VAD events
                    print(f"VAD event: {event} - {message}")
                    
                    # If speech ended, this audio could be sent for STT processing
                    if event == "end":
                        self.stt_queue.put(message)
            except grpc.RpcError as e:
                print(f"RPC error: {e}")
            except Exception as e:
                print(f"Error processing VAD responses: {e}")
            
            print("Finished processing file")
    
    def stop_streaming(self):
        """Stop streaming audio"""
        self.running = False
    
    def reset_vad(self):
        """Reset the VAD state on the server"""
        with grpc.insecure_channel(self.server_address) as channel:
            stub = vad_pb2_grpc.VADServiceStub(channel)
            response = stub.ResetVAD(vad_pb2.ResetRequest())
            return response.success
    
    def run_in_background(self):
        """Run VAD streaming in a background thread"""
        thread = threading.Thread(target=self.stream_file)
        thread.daemon = True
        thread.start()
        return thread


# Example usage with a mock STT processor
class STTProcessor:
    def __init__(self, vad_client):
        self.vad_client = vad_client
        self.running = False
    
    def process_stt_queue(self):
        """Process the STT queue when VAD detects end of speech"""
        self.running = True
        while self.running:
            try:
                message = self.vad_client.stt_queue.get(timeout=1)
                print(f"Processing STT: {message}")
                # In a real application, you would send this audio to your STT service
                # stt_result = your_stt_service.process(audio_data)
                # print(f"STT result: {stt_result}")
            except queue.Empty:
                if not self.vad_client.running:
                    # If the client is no longer running and queue is empty, exit
                    break
            except Exception as e:
                print(f"Error processing STT: {e}")
    
    def run_in_background(self):
        """Run STT processing in a background thread"""
        thread = threading.Thread(target=self.process_stt_queue)
        thread.daemon = True
        thread.start()
        return thread
    
    def stop(self):
        """Stop STT processing"""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description='VAD Client for WAV files')
    parser.add_argument('--server', default='localhost:50055', help='gRPC server address')
    parser.add_argument('--file', required=True, help='Path to WAV file')
    
    args = parser.parse_args()
    
    try:
        # Create and start VAD client
        vad_client = FileVADClient(server_address=args.server, wav_file=args.file)
        vad_thread = vad_client.run_in_background()
        
        # Create and start STT processor
        stt_processor = STTProcessor(vad_client)
        stt_thread = stt_processor.run_in_background()
        
        print(f"Processing file: {args.file}")
        print("Press Ctrl+C to stop.")
        
        # Wait for threads to complete
        vad_thread.join()
        stt_thread.join()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'vad_client' in locals():
            vad_client.stop_streaming()
        if 'stt_processor' in locals():
            stt_processor.stop()


if __name__ == "__main__":
    main()
