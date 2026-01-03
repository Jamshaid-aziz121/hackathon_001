#!/usr/bin/env python3
"""
Multimodal Reasoning Example for Educational Robotics

This example demonstrates how to combine visual and language inputs
to perform reasoning tasks in an educational context.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
import openai
import os
from typing import Dict, List, Tuple, Any
import json

class MultimodalReasoning:
    def __init__(self):
        """Initialize multimodal reasoning components"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize CLIP for vision-language understanding
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            self.clip_model = None

        # Initialize language model for reasoning
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.language_model.to(self.device)
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("GPT-2 model loaded successfully")
        except Exception as e:
            print(f"Failed to load GPT-2 model: {e}")
            self.language_model = None

        # Define educational objects and concepts
        self.educational_objects = [
            'book', 'pencil', 'paper', 'calculator', 'ruler',
            'student', 'teacher', 'desk', 'chair', 'whiteboard',
            'ball', 'toy', 'blocks', 'shapes', 'letters', 'numbers',
            'robot', 'computer', 'laptop', 'tablet'
        ]

        # Define spatial relations
        self.spatial_relations = [
            'left of', 'right of', 'in front of', 'behind',
            'near', 'far from', 'on top of', 'under', 'next to'
        ]

    def process_visual_input(self, image_path: str) -> Dict[str, Any]:
        """Process visual input using CLIP"""
        if self.clip_model is None:
            return {'objects': [], 'features': None}

        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Get image features
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**inputs)

            # Generate text prompts for educational objects
            text_prompts = [f"a photo of a {obj}" for obj in self.educational_objects]
            text_inputs = self.clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)

            # Calculate similarities
            logits_per_image = self.clip_model.get_image_text_features(**inputs, input_ids=text_inputs.input_ids)
            probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()[0]

            # Get top predictions
            top_indices = np.argsort(probs)[::-1][:5]
            detected_objects = []

            for idx in top_indices:
                confidence = probs[idx]
                if confidence > 0.1:  # Threshold for confidence
                    detected_objects.append({
                        'object': self.educational_objects[idx],
                        'confidence': float(confidence)
                    })

            return {
                'objects': detected_objects,
                'features': image_features.cpu().detach().numpy(),
                'image_size': image.size
            }
        except Exception as e:
            print(f"Error processing visual input: {e}")
            return {'objects': [], 'features': None}

    def process_language_input(self, text: str) -> Dict[str, Any]:
        """Process language input for reasoning"""
        if self.language_model is None:
            return {'entities': [], 'intent': 'unknown'}

        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Generate continuation for reasoning
            with torch.no_grad():
                outputs = self.language_model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 20,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract entities and intent (simplified approach)
            entities = self.extract_entities(text)
            intent = self.classify_intent(text)

            return {
                'original_text': text,
                'generated_text': generated_text,
                'entities': entities,
                'intent': intent,
                'tokens': inputs.input_ids.shape[1]
            }
        except Exception as e:
            print(f"Error processing language input: {e}")
            return {'entities': [], 'intent': 'unknown'}

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        entities = []
        text_lower = text.lower()

        # Look for educational objects in text
        for obj in self.educational_objects:
            if obj in text_lower:
                entities.append(obj)

        # Look for spatial relations
        for relation in self.spatial_relations:
            if relation in text_lower:
                entities.append(relation)

        return entities

    def classify_intent(self, text: str) -> str:
        """Classify intent of the text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['find', 'look', 'see', 'where', 'locate']):
            return 'find_object'
        elif any(word in text_lower for word in ['move', 'go', 'walk', 'navigate', 'approach']):
            return 'navigation'
        elif any(word in text_lower for word in ['count', 'how many', 'number', 'total']):
            return 'counting'
        elif any(word in text_lower for word in ['describe', 'what', 'explain', 'tell me']):
            return 'description'
        elif any(word in text_lower for word in ['compare', 'difference', 'similar', 'different']):
            return 'comparison'
        else:
            return 'other'

    def perform_multimodal_reasoning(self, image_path: str, question: str) -> Dict[str, Any]:
        """Perform multimodal reasoning combining vision and language"""
        # Process visual input
        visual_result = self.process_visual_input(image_path)

        # Process language input
        language_result = self.process_language_input(question)

        # Combine information for reasoning
        reasoning_result = {
            'question': question,
            'detected_objects': visual_result['objects'],
            'language_analysis': language_result,
            'reasoning_output': '',
            'confidence': 0.0
        }

        # Perform specific reasoning based on intent
        intent = language_result['intent']

        if intent == 'find_object':
            reasoning_result['reasoning_output'] = self.reason_about_object_location(
                visual_result['objects'], question
            )
        elif intent == 'counting':
            reasoning_result['reasoning_output'] = self.reason_about_counting(
                visual_result['objects'], question
            )
        elif intent == 'description':
            reasoning_result['reasoning_output'] = self.reason_about_description(
                visual_result['objects'], question
            )
        elif intent == 'comparison':
            reasoning_result['reasoning_output'] = self.reason_about_comparison(
                visual_result['objects'], question
            )
        else:
            reasoning_result['reasoning_output'] = self.general_reasoning(
                visual_result['objects'], question
            )

        # Calculate overall confidence
        if visual_result['objects']:
            avg_confidence = sum([obj['confidence'] for obj in visual_result['objects']]) / len(visual_result['objects'])
            reasoning_result['confidence'] = avg_confidence
        else:
            reasoning_result['confidence'] = 0.1  # Low confidence if no objects detected

        return reasoning_result

    def reason_about_object_location(self, detected_objects: List[Dict], question: str) -> str:
        """Reason about object location based on question"""
        target_objects = []
        for obj in detected_objects:
            if obj['object'] in question.lower():
                target_objects.append(obj)

        if target_objects:
            if len(target_objects) == 1:
                obj = target_objects[0]
                return f"I found the {obj['object']} with confidence {obj['confidence']:.2f}."
            else:
                obj_names = [obj['object'] for obj in target_objects]
                return f"I found multiple objects: {', '.join(obj_names)}."
        else:
            obj_names = [obj['object'] for obj in detected_objects]
            if obj_names:
                return f"I don't see the specific object you mentioned, but I can see: {', '.join(obj_names)}."
            else:
                return "I don't see any recognizable objects in the image."

    def reason_about_counting(self, detected_objects: List[Dict], question: str) -> str:
        """Reason about counting objects"""
        # Count all detected objects
        total_count = len(detected_objects)

        # If looking for specific object type
        for obj in self.educational_objects:
            if obj in question.lower():
                specific_count = sum(1 for det_obj in detected_objects if det_obj['object'] == obj)
                return f"I see {specific_count} {obj}(s) with reasonable confidence."

        return f"I see approximately {total_count} objects total."

    def reason_about_description(self, detected_objects: List[Dict], question: str) -> str:
        """Reason about describing the scene"""
        if not detected_objects:
            return "The image appears to be empty or I cannot recognize the objects in it."

        # Sort by confidence
        sorted_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)
        top_objects = sorted_objects[:3]  # Top 3 most confident detections

        object_names = [f"{obj['object']} (confidence: {obj['confidence']:.2f})" for obj in top_objects]

        return f"In the image, I can see: {', '.join(object_names)}. This appears to be an educational environment."

    def reason_about_comparison(self, detected_objects: List[Dict], question: str) -> str:
        """Reason about comparing objects"""
        if len(detected_objects) < 2:
            return "I need to see at least two objects to make a comparison."

        # For now, just list the objects
        object_names = [obj['object'] for obj in detected_objects]
        return f"I can see these objects: {', '.join(object_names)}. I can help compare them if you specify what you'd like to compare."

    def general_reasoning(self, detected_objects: List[Dict], question: str) -> str:
        """General reasoning when specific intent not identified"""
        if detected_objects:
            top_obj = max(detected_objects, key=lambda x: x['confidence'])
            return f"Based on the image, I see a {top_obj['object']} with confidence {top_obj['confidence']:.2f}. {question}"
        else:
            return "I cannot identify objects in the image to answer your question."

    def generate_educational_response(self, reasoning_result: Dict[str, Any]) -> str:
        """Generate an educational response based on reasoning"""
        base_response = reasoning_result['reasoning_output']

        # Add educational context based on detected objects
        detected_obj_names = [obj['object'] for obj in reasoning_result['detected_objects']]

        educational_additions = []

        if 'book' in detected_obj_names:
            educational_additions.append("This looks like a learning environment with educational materials.")

        if 'student' in detected_obj_names or 'teacher' in detected_obj_names:
            educational_additions.append("This appears to be a classroom setting.")

        if 'robot' in detected_obj_names:
            educational_additions.append("Robots can be great tools for learning science and technology concepts!")

        if educational_additions:
            return base_response + " " + " ".join(educational_additions)
        else:
            return base_response

    def run_example(self):
        """Run a complete multimodal reasoning example"""
        print("Running Multimodal Reasoning Example for Educational Robotics")

        # Create a sample image (in practice, this would be a real image file)
        # For this example, we'll simulate by using a text-based approach
        sample_questions = [
            "What do you see in the image?",
            "Find the book in the image",
            "How many objects are there?",
            "Describe the educational environment"
        ]

        print("\nSample reasoning outputs:")
        for i, question in enumerate(sample_questions, 1):
            print(f"\nQuestion {i}: {question}")

            # In a real implementation, you would process an actual image
            # For this example, we'll simulate with a mock result
            mock_objects = [
                {'object': 'book', 'confidence': 0.85},
                {'object': 'student', 'confidence': 0.78},
                {'object': 'desk', 'confidence': 0.72}
            ]

            mock_reasoning = {
                'question': question,
                'detected_objects': mock_objects,
                'language_analysis': {'intent': self.classify_intent(question)},
                'reasoning_output': f"Simulated response to: {question}",
                'confidence': 0.8
            }

            educational_response = self.generate_educational_response(mock_reasoning)
            print(f"Response: {educational_response}")
            print(f"Confidence: {mock_reasoning['confidence']:.2f}")

def create_sample_image():
    """Create a sample educational image for testing"""
    # Create a simple image with shapes representing educational objects
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw some educational objects
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle (book)
    cv2.circle(img, (300, 100), 40, (0, 255, 0), -1)  # Green circle (ball)
    cv2.rectangle(img, (400, 50), (550, 200), (0, 0, 255), -1)  # Red rectangle (desk)

    # Add text labels
    cv2.putText(img, 'Book', (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, 'Ball', (275, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, 'Desk', (405, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save the image
    cv2.imwrite('sample_educational_scene.jpg', img)
    print("Sample educational scene image created: sample_educational_scene.jpg")

def main():
    """Main function to run the multimodal reasoning example"""
    print("Initializing Multimodal Reasoning System for Educational Robotics")

    # Create a sample image for the example
    create_sample_image()

    # Initialize the multimodal reasoning system
    mm_reasoner = MultimodalReasoning()

    # Run the example
    mm_reasoner.run_example()

    print("\nFor real applications:")
    print("1. Replace sample image with actual educational environment images")
    print("2. Use the perform_multimodal_reasoning() method with real images and questions")
    print("3. Integrate with robot control systems for physical interaction")

if __name__ == "__main__":
    main()