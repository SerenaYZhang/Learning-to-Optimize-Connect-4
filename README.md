# Learning to Optimize Connect 4 Play Using Heuristic Search & A Trained Neural Network

**Team Members**: Ryan Park (ryp3), Nicole Sin (ns753), Serena Zhang (syz8)

## Description

We propose to develop a Connect 4 system that uses an AI to play optimally. A GUI will be used for the 7x6 Connect 4 board so that the two players can consist of human users, the heuristic approach AI, or the neural network AI.

Our heuristic AI will prioritize three main objectives:
1. **Defensive play** - blocking an opponent's three-in-a-row to prevent losses
2. **Offensive play** - creating opportunities to achieve four-in-a-row
3. **Positional control** - favoring the center column, which statistically increases winning chances

After implementing the heuristics, we will train a neural network on a Kaggle dataset of Connect 4 games, which provides numerous final board positions and outcomes. By learning from these examples, the AI can infer winning strategies, recognize patterns in play, and prioritize moves that maximize its chance of winning.

The final deliverable will be a fully interactive system where users can choose to test their skills against the AI, or the heuristic will play against the neural network approach. The GUI will also display statistics, such as the win-rate that aligns with near-optimal play, allowing us to quantify how well each AI approach performs.

## General Approach

Our first approach will mainly utilize heuristic search. Using a minimax search algorithm, we will explore possible future board states to find the best move for a player (assuming that the other person plays optimally as well). The way to reduce the number of moves the AI has to comb through is by implementing alpha-beta pruning, which eliminates many possible branches that are not relevant to the model, which can drastically increase the computational efficiency of our model.

We will also implement a neural network and train it on our Kaggle dataset, and use that as an evaluation method to compare the win-rates of our heuristic search algorithm to the neural network. From this feedback, we can adjust our algorithm to weigh certain heuristics more favorably (or decrease their weight) to achieve a closer optimal play.

If time allows, we hope to develop a model that iterates through game positions to find the optimal play of situations and use that as an additional comparison tool to evaluate the performance of either the heuristic model or the trained neural network.

## Plan for Evaluation

We will evaluate our system in the following methods:

- **Optimality comparison**: Measure the win rate. If we have time, we can measure the percentage of moves that match known optimal plays.
- **Human testing**: Allow users to play against the AI through the GUI and gather feedback on perceived difficulty and realism of play.
- **Model Evaluation**: Compare the heuristic approach against the neural network approach
- **Model Changes**: Based on performance comparisons, we plan to change weights in our model to optimize performance.

Additionally, we will track whether incorporating games generated against our AI improves future play performance, testing the system's ability to learn iteratively.

## Timeline

| Date | Milestone |
|------|-----------|
| 10/03 | Submit project proposal |
| 10/20 | Build the GUI (7×6 Connect 4 board) and backend logic |
| 11/07 | Implement heuristic search (minimax + alpha–beta pruning) |
| 11/21 | Train and integrate the neural network model using the Kaggle dataset |
| 12/05 | Prepare report, demo, presentation, and video synopsis |
| Final Week | Perform corrections, polish GUI, and finalize results |

## Task Distribution

| Team Member | Responsibilities |
|-------------|------------------|
| Ryan Park | GUI design, user interaction, evaluation metrics |
| Nicole Sin | Heuristic search implementation (minimax + alpha–beta pruning) |
| Serena Zhang | Neural network training, dataset management, and integration |