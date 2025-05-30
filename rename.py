import os

def rename_files(directory, old_pattern, new_pattern):
    """Renames files in a directory based on a pattern."""

    for filename in os.listdir(directory):
        if old_pattern in filename:
            new_filename = filename.replace(old_pattern, new_pattern)
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

if __name__ == "__main__":
    directory = "./data/coatings/manual-measurements"  # Replace with your actual directory
    old_pattern = "CV_0_"  # Replace with the text you want to replace
    new_pattern = "CV_"  # Replace with the new text

    rename_files(directory, old_pattern, new_pattern)