// static/audio-processor.js

class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
      super();
      // Define the target buffer size (number of samples per chunk to send)
      this.targetBufferSize = 512;
      // Buffer to accumulate floating-point audio data
      this.buffer = new Float32Array(this.targetBufferSize);
      // Current position (number of samples) in the buffer
      this.bufferPos = 0;
  
      // Optional: Handle messages from the main thread (e.g., for flushing)
      // this.port.onmessage = (event) => {
      //   if (event.data === 'flush') {
      //     this.flush();
      //   }
      // };
    }
  
    // Helper function to convert and send the buffer
    sendBuffer() {
      // Ensure we actually have data to send (might be slightly less than targetBufferSize on flush)
       if (this.bufferPos === 0) {
           return;
       }
  
      // Note: If flushing is implemented and bufferPos < targetBufferSize,
      // you might want to decide whether to send a partial buffer
      // or pad it with zeros. For now, we only send full buffers.
      // Let's create the Int16Array with the actual size accumulated.
      // For the requirement of STRICTLY 512 samples, we ensure it only runs when full.
  
      // Convert the accumulated Float32Array data to Int16Array
      const int16Buffer = new Int16Array(this.targetBufferSize); // Always create size 512
      for (let i = 0; i < this.targetBufferSize; i++) {
        const sample = Math.max(-1, Math.min(1, this.buffer[i])); // Clamp
        int16Buffer[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF; // Convert
      }
  
      // Post the Int16Array buffer back to the main thread.
      // Transfer ownership for efficiency.
      this.port.postMessage(int16Buffer, [int16Buffer.buffer]);
  
      // Reset the buffer position
      this.bufferPos = 0;
    }
  
    // // Optional: Flush remaining data (e.g., when stopping)
    // flush() {
    //   if (this.bufferPos > 0) {
    //     // Decide how to handle partial buffer: send as is, or pad?
    //     // Sending as is (might require server adjustments):
    //     const remainingData = this.buffer.slice(0, this.bufferPos);
    //     const int16Buffer = new Int16Array(remainingData.length);
    //     for (let i = 0; i < remainingData.length; i++) {
    //          const sample = Math.max(-1, Math.min(1, remainingData[i]));
    //          int16Buffer[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    //      }
    //     this.port.postMessage(int16Buffer, [int16Buffer.buffer]);
    //     this.bufferPos = 0;
    //   }
    // }
  
    process(inputs, outputs, parameters) {
      // We only process the first input and first channel
      const inputChannelData = inputs[0]?.[0]; // Use optional chaining for safety
  
      // If no input data, just keep processor alive
      if (!inputChannelData || inputChannelData.length === 0) {
        return true;
      }
  
      // Append the incoming data (typically 128 samples) to our buffer
      let currentInputPos = 0;
      while (currentInputPos < inputChannelData.length) {
          // How many samples can we copy into the buffer this time?
          const remainingInBuffer = this.targetBufferSize - this.bufferPos;
          const samplesToCopy = Math.min(inputChannelData.length - currentInputPos, remainingInBuffer);
  
          // Copy the data
          const sliceToCopy = inputChannelData.subarray(currentInputPos, currentInputPos + samplesToCopy);
          this.buffer.set(sliceToCopy, this.bufferPos);
  
          // Update positions
          this.bufferPos += samplesToCopy;
          currentInputPos += samplesToCopy;
  
          // If the buffer is full, process and send it
          if (this.bufferPos >= this.targetBufferSize) {
              this.sendBuffer(); // This also resets bufferPos to 0
          }
      }
  
  
      // Return true to keep the processor alive
      return true;
    }
  }
  
  registerProcessor('audio-processor', AudioProcessor);