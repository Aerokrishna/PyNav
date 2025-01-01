import openai
import re

class InferHumanCommand():
    def __init__(self):
        # Define hard-coded locations with their coordinates
        self.locations = {
            
            "Table 1": (2.0, 3.5),
            "Table 2": (2.1, 1.3),
            "Table 3": (4.0, 1.5),
            "Table 4": (4.1, 3.5),
            "Kitchen": (6.0, 4.0)
        }

    def infer_human_command(self):
        # Initialize OpenAI API
        openai.api_key = ""

        # Input natural language command
        user_input = input("Enter your navigation command: ")

        # Use OpenAI GPT API to parse and convert to location names
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a robot navigation assistant. Convert the given natural language navigation command into a list of waypoints using location names like 'Table A', 'Kitchen', and 'Table B'."},
                {"role": "user", "content": user_input}
            ]
        )

        # Extract the structured response
        structured_output = response['choices'][0]['message']['content']
        print("Structured Output:\n", structured_output)

        return structured_output

    def parse_waypoints(self, output):
        waypoints = []
        # Extract locations mentioned in the output
        for location in self.locations.keys():
            if location in output:
                waypoints.append(self.locations[location])
        return waypoints

# # Instantiate the class
# navigator = InferHumanCommand()

# # Convert user input to structured output
# structured_output = navigator.infer_human_command()

# # Parse the structured output to get waypoints
# goal_points = navigator.parse_waypoints(structured_output)
# print("Goal Points:", goal_points)
