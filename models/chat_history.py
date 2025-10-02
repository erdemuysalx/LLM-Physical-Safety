import os


class ChatHistory:
    def __init__(self, system_prompt=None):
        if system_prompt is None:
            file_path = os.path.abspath(__file__)
            parent_dir = os.path.join(os.path.dirname(file_path), '..')
            file_path = os.path.join(parent_dir, 'system_prompts', 'airsim_extension.txt')
            with open(file_path, 'r') as file:
                system_prompt = file.read()
            self.system_prompt = system_prompt
        else:
            self.system_prompt = system_prompt
        self.q1 = "move 10 units up"
        self.a1 = "```python\n" \
                  "aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])\n" \
                  "```\n\n" \
                  "This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position." \
                  " It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates," \
                  " but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."

        self.openai = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.q1
            },
            {
                "role": "assistant",
                "content": self.a1
            }
        ]

        self.gemini = [
            {
                'role': 'user',
                'parts': [self.system_prompt]
            },
            {
                'role': 'model',
                'parts': ["No problem."]
            },
            {
                'role': 'user',
                'parts': [self.q1]
            },
            {
                'role': 'model',
                'parts': [self.a1]
            }
        ]

