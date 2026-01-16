import json
import os


class OutputWriter:
    """
    This class is responsible for storing and writing information about people to a txt file.
    Each person is identified by a unique ID and their information includes gender, hat, bag
    and their trajectory (the ordered sequence of lines they have crossed).
    """

    def __init__(self):
        """
        Initializes the OutputWriter by creating an empty list to store the information of people.
        """
        self.people = []  # List to store the people data

    def add_person(self, person_id, gender, bag, hat, trajectory):
        """
        Adds a new person to the people list.

        Parameters:
        person_id : int
            Unique identifier for the person
        gender : int
            Gender of the person, where 0 represents male and 1 represents female
        bag : int
            Whether the person is carrying a bag
        hat : int
            Whether the person is wearing a hat
        trajectory : lis
            List of virtual line IDs representing the ordered sequence of lines crossed by the person
        """
        # Check if the person already exists in the list by their ID
        for person in self.people:
            if person["id"] == person_id:
                print(f"Person with ID {person_id} already exists.")
                return

        # Convert gender from integer (0 or 1) to string ("male" or "female")
        gender = "male" if gender == 0 else "female"

        # Convert bag and hat to boolean values
        bag = bool(bag)
        hat = bool(hat)

        # trajectory = "[" + ",".join(str(x) for x in trajectory) + "]"

        # Create a dictionary for the new person with their details
        person = {
            "id": person_id,  # Person's ID
            "gender": gender,  # Person's gender
            "hat": hat,  # Whether the person is wearing a hat
            "bag": bag,  # Whether the person is carrying a bag
            "trajectory": trajectory  # List of lines crossed by the person
        }

        # Append the new person to the people list
        self.people.append(person)

    def write_output(self, filename="./result/result.txt"):
        """
        Writes the information about all people stored in the list to a file.

        Parameters:
        filename : str
            The path and name of the output file (default is "./result/result.txt").
        """
        # Extract the directory from the filename
        directory = os.path.dirname(filename)

        # Create the directory if it does not exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Create the output dictionary with the list of people
        output = {"people": self.people}

        # Open the file in write mode and write the data to it in JSON format with indentation
        with open(filename, 'w+') as file:
            json.dump(output, file, indent=4)
