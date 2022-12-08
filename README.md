# Predict Question Answered
A TensorFlow neural network that evaluates the effectiveness of Stack Overflow questions using metadata. Given the Title Length, Total Word Count, Number of Code Snippets, Total Code Length, Number of Images, and Number of Tags, this model can predict if the question will receive an accepted answer with ~85% accuracy.

## Setup Instructions
1. Run `./dependencies.sh` to install the required dependencies
2. Download a subset of the [Stack Overflow dataset](https://ia600107.us.archive.org/27/items/stackexchange/Stackoverflow.com-Posts.7z) from [Google Drive](https://drive.google.com/file/d/1FMlo6lFDQJ3Sw13VABGFde-VyX1lKBnL/view?usp=share_link)
3. Upzip the file and move it to the `data/` directory

## Run Instructions
1. Run `python model.py` to generate a model
2. Run `python predict.py [model_filename]` to run predictions using sample data
2. Run `python importance.py [model_filename]` to see the importance of each metadata parameter

## Team Members
- Cameron LaBounty
- Ben Roberts
- Rhys Sullivan
- William Kloppenberg