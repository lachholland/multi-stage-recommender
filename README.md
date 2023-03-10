<h1>Multi-Stage Recommendation System</h1>

A multi-stage recommendation system built using PyTorch trained on data sourced from the H&M Personalized Fashion Recommendations Kaggle competition.


### How to run

1. Activate the python virtual environment. From the project root folder, run the terminal command:
   ```sh
   venv/bin/activate
   ```
2. Install project dependencies by running the makefile. From the project root folder, run the terminal command:
   ```sh
   make
   ```
3. Train the model. From the /src folder run the terminal command:
   ```sh
   python -m main
   ```
   

## toDos

- Add negative sampling
- Add hyperparameters
- Add additional user and item features


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Model training data sourced from the Kaggle competition: [H&M Personalized Fashion Recommendations Kaggle competition](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations).
Inspiration from the blog post by Adrien Biarnes: [Building a Multi-Stage Recommendation System (Part 1.1)](https://medium.com/mlearning-ai/building-a-multi-stage-recommendation-system-part-1-1-95961ccf3dd8).

