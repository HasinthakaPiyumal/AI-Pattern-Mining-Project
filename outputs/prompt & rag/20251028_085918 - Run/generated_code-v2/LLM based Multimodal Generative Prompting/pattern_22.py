from PIL import Image
import io
import random

class ImageGenerationService:
    def __init__(self):
        pass

    def generate_image(self, positive_prompt: str, negative_prompt: str) -> dict:
        # This is a conceptual implementation. In a real application,
        # this would interact with a powerful AI image generation model
        # like Stable Diffusion via a library like 'diffusers'.
        # The negative_prompt would be passed directly to the model's generation pipeline.

        print(f"\n--- Simulating Image Generation ---")
        print(f"Positive Prompt: '{positive_prompt}'")
        print(f"Negative Prompt: '{negative_prompt}'")
        print(f"Leveraging AI model to generate an image based on '{positive_prompt}' while avoiding '{negative_prompt}'.")

        # Simulate generating a blank image for demonstration purposes
        width, height = 512, 512
        img = Image.new('RGB', (width, height), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

        # In a real scenario, the model would return an image object or bytes
        # For this example, we'll save it to a dummy path or convert to bytes
        # to simulate output.
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Simulate a filename for the generated image
        output_filename = f"generated_product_design_{hash(positive_prompt + negative_prompt)}.png"
        # You could save the image here:
        # img.save(output_filename)

        return {
            "status": "success",
            "message": f"Image generated successfully, conceptually saved as {output_filename}",
            "image_data": img_byte_arr.getvalue() # In a real app, this might be a URL or direct image data
        }

class ECommerceFrontend:
    def __init__(self):
        self.image_service = ImageGenerationService()
        self.default_negative_prompt = "blurry, distorted, text, watermark, discolored, multiple heads, too many limbs, low quality, bad anatomy, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad art, grainy, signature, cut off, draft"

    def run(self):
        print("\n--- E-commerce Product Customizer ---")
        print("Design your custom product using AI image generation.")
        print("We'll automatically apply negative prompting to ensure high quality and avoid common issues.")

        while True:
            positive_prompt = input("\nEnter your desired design (e.g., 'a majestic dragon flying over mountains'): ")
            if not positive_prompt.strip():
                print("Please enter a design prompt.")
                continue

            # Simulate product type selection - not used in image gen for this example
            # product_type = input("Enter product type (e.g., 't-shirt', 'mug') (optional): ")

            generation_result = self.image_service.generate_image(
                positive_prompt=positive_prompt,
                negative_prompt=self.default_negative_prompt
            )

            if generation_result["status"] == "success":
                print(generation_result["message"])
                # In a real application, you would display the image here.
                # For this conceptual example, we just confirm generation.
                print("Image data received (conceptual visual for the user).")
            else:
                print(f"Error generating image: {generation_result['message']}")

            another = input("Generate another design? (yes/no): ").lower()
            if another != 'yes':
                break

        print("\nThank you for using the E-commerce Product Customizer!")

if __name__ == "__main__":
    frontend = ECommerceFrontend()
    frontend.run()