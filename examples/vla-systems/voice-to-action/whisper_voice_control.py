#!/usr/bin/env python3
"""
OpenAI Whisper Integration Example for Voice-to-Action in Educational Robotics

This example demonstrates how to integrate OpenAI Whisper for speech recognition
with robotic action execution in an educational context.
"""

import os
import time
import numpy as np
import torch
import pyaudio
import wave
import openai
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from PIL import Image as PILImage

class WhisperVoiceControl:
    def __init__(self):
        """Initialize the Whisper voice control system"""
        # Initialize Whisper model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Whisper model (using smaller model for efficiency)
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
            self.model.to(self.device)
            self.model.eval()
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            # Fallback to a simple speech recognition approach
            self.model = None

        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best at 16kHz
        self.record_seconds = 5

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Robot control publisher (simulated in this example)
        self.robot_cmd_pub = None  # Would be initialized in ROS context

        # Define command mappings for educational robotics
        self.command_map = {
            'move forward': 'forward',
            'go forward': 'forward',
            'move backward': 'backward',
            'go backward': 'backward',
            'turn left': 'left',
            'turn right': 'right',
            'stop': 'stop',
            'find object': 'find_object',
            'look for object': 'find_object',
            'help me': 'help',
            'teach me': 'teach',
            'show me': 'show',
            'what is this': 'identify',
            'explain this': 'explain'
        }

        # Confidence threshold for commands
        self.confidence_threshold = 0.7

    def record_audio(self, filename="temp_audio.wav"):
        """Record audio from microphone"""
        print(f"Recording audio for {self.record_seconds} seconds...")

        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished")

        stream.stop_stream()
        stream.close()

        # Save the recorded audio
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return filename

    def transcribe_with_whisper(self, audio_file_path):
        """Transcribe audio using Whisper model"""
        if self.model is None:
            return "Speech recognition not available"

        try:
            # Load and preprocess audio
            import librosa
            audio, sr = librosa.load(audio_file_path, sr=16000)

            # Process with Whisper
            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)

            # Generate token ids
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            # Decode token ids to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            print(f"Error in Whisper transcription: {e}")
            return ""

    def simple_speech_to_text(self, audio_file_path):
        """Fallback simple speech to text (placeholder)"""
        # In a real implementation, you might use a simpler model or API
        # For this example, we'll simulate recognition
        print(f"Using fallback speech recognition for {audio_file_path}")
        return "go forward"  # Placeholder response

    def recognize_speech(self):
        """Complete speech recognition pipeline"""
        # Record audio
        audio_file = self.record_audio()

        # Transcribe using Whisper or fallback
        if self.model is not None:
            transcription = self.transcribe_with_whisper(audio_file)
        else:
            transcription = self.simple_speech_to_text(audio_file)

        print(f"Recognized: {transcription}")

        # Clean up temporary file
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return transcription

    def parse_command(self, text):
        """Parse recognized text into robot command"""
        text_lower = text.lower().strip()

        # Check for exact matches first
        if text_lower in self.command_map:
            return self.command_map[text_lower]

        # Check for partial matches
        for key, value in self.command_map.items():
            if key in text_lower:
                return value

        # If no match found, return unknown
        return 'unknown'

    def execute_command(self, command):
        """Execute the parsed command"""
        print(f"Executing command: {command}")

        if command == 'forward':
            self.move_forward()
        elif command == 'backward':
            self.move_backward()
        elif command == 'left':
            self.turn_left()
        elif command == 'right':
            self.turn_right()
        elif command == 'stop':
            self.stop_robot()
        elif command == 'find_object':
            self.find_object()
        elif command == 'help':
            self.provide_help()
        elif command == 'teach':
            self.start_teaching_mode()
        elif command == 'show':
            self.show_demonstration()
        elif command == 'identify':
            self.identify_object()
        elif command == 'explain':
            self.explain_concept()
        else:
            print(f"Unknown command: {command}")
            self.speak_response("I don't understand that command. Please try again.")

    def move_forward(self):
        """Move robot forward"""
        print("Moving robot forward")
        # In ROS context: publish Twist message with linear.x > 0
        if self.robot_cmd_pub:
            cmd = Twist()
            cmd.linear.x = 0.3  # Move forward at 0.3 m/s
            self.robot_cmd_pub.publish(cmd)

    def move_backward(self):
        """Move robot backward"""
        print("Moving robot backward")
        if self.robot_cmd_pub:
            cmd = Twist()
            cmd.linear.x = -0.3  # Move backward at 0.3 m/s
            self.robot_cmd_pub.publish(cmd)

    def turn_left(self):
        """Turn robot left"""
        print("Turning robot left")
        if self.robot_cmd_pub:
            cmd = Twist()
            cmd.angular.z = 0.5  # Turn left at 0.5 rad/s
            self.robot_cmd_pub.publish(cmd)

    def turn_right(self):
        """Turn robot right"""
        print("Turning robot right")
        if self.robot_cmd_pub:
            cmd = Twist()
            cmd.angular.z = -0.5  # Turn right at 0.5 rad/s
            self.robot_cmd_pub.publish(cmd)

    def stop_robot(self):
        """Stop robot movement"""
        print("Stopping robot")
        if self.robot_cmd_pub:
            cmd = Twist()
            self.robot_cmd_pub.publish(cmd)  # Zero velocity stops the robot

    def find_object(self):
        """Initiate object finding behavior"""
        print("Looking for objects...")
        # In a real implementation, this would activate vision processing
        self.speak_response("I'm looking for objects. Please wait.")

    def provide_help(self):
        """Provide help to the user"""
        print("Providing help")
        self.speak_response("I'm here to help! You can tell me to move forward, backward, turn left or right, or ask me to find objects.")

    def start_teaching_mode(self):
        """Start teaching mode"""
        print("Starting teaching mode")
        self.speak_response("Entering teaching mode. What would you like to learn about?")

    def show_demonstration(self):
        """Show a demonstration"""
        print("Showing demonstration")
        self.speak_response("I can demonstrate various educational concepts. What would you like to see?")

    def identify_object(self):
        """Identify objects in the environment"""
        print("Identifying objects")
        self.speak_response("I'm analyzing the objects around us.")

    def explain_concept(self):
        """Explain a concept"""
        print("Explaining concept")
        self.speak_response("I can explain various educational concepts. What would you like to know?")

    def speak_response(self, text):
        """Speak response (simulated)"""
        print(f"Robot says: {text}")
        # In a real implementation, this would use text-to-speech

    def run_continuous_listening(self):
        """Run continuous listening loop"""
        print("Starting continuous listening mode. Press Ctrl+C to stop.")

        try:
            while True:
                print("\nListening for commands...")

                # Recognize speech
                recognized_text = self.recognize_speech()

                if recognized_text:
                    # Parse command
                    command = self.parse_command(recognized_text)

                    if command != 'unknown':
                        # Execute command
                        self.execute_command(command)
                    else:
                        print(f"Could not understand command: {recognized_text}")
                        self.speak_response("I didn't understand that. Could you repeat it?")

                # Small delay before next iteration
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping voice control system...")
        finally:
            # Clean up audio resources
            self.audio.terminate()

def main():
    """Main function to run the Whisper voice control example"""
    print("Initializing Whisper Voice Control for Educational Robotics")

    # Create voice control instance
    voice_control = WhisperVoiceControl()

    print("Whisper Voice Control initialized successfully!")
    print("Available commands: move forward, go backward, turn left, turn right, stop, find object, help me, teach me, show me, what is this, explain this")

    # Run continuous listening
    voice_control.run_continuous_listening()

if __name__ == "__main__":
    main()