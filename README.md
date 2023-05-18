# PyTorch Transformer Implementation: Detailed from Scratch

## Introduction

This repository contains a comprehensive, from-scratch implementation of the Transformer model using PyTorch. The Transformer model, a groundbreaking architecture introduced in the seminal paper "Attention is All You Need", has revolutionized the field of natural language processing. Here, we provide two unique versions of this model, including a complete debugging notebook, and a use-case for English-Portuguese translation.

## Table of Contents
1. [Visualization](#visualization)
2. [Project Structure](#project-structure)
3. [Contributing](#contributing)
4. [License](#license)
5. [Contact Information](#contact-information)

## Visualization

The `debug_notebook.ipynb` file is instrumental in visualizing key aspects of the Transformer architecture:

- **Weights**: Visualizing the weights allows us to understand how the model changes them during training, which can give insights into the learning process.

![Transformer Weights](https://github.com/reiid00/pytorch-transformer-implementation/blob/master/Imgs/Attention_weights.png)



- **Positional Encodings**: These are crucial in the Transformer model as they provide a sense of order of the input to the model, which lacks recurrence and convolution.

![Pos Encodings](https://github.com/reiid00/pytorch-transformer-implementation/blob/master/Imgs/pos_encodings.png)



- **Learning Rate Decay**: Plotting the learning rate decay helps in understanding how the learning rate changes over time or epochs, which can be beneficial for hyperparameter tuning.

![LR Decay](https://github.com/reiid00/pytorch-transformer-implementation/blob/master/Imgs/lr_decay.png)



- **Gradients**: Visualizing the gradients can help detect if they are vanishing or exploding, a common issue in deep learning models.

![Gradients](https://github.com/reiid00/pytorch-transformer-implementation/blob/master/Imgs/gradient_flow.png)




## Project Structure

This project comprises an extensive set of files and directories, each with a distinct purpose:

- `Translation-PT-EN`: This directory serves as a practical use-case for the Transformer model, focusing on Portuguese-English translation. It includes scripts for data loading, model training, and greedy decoding. Due to hardware constraints, results are not perfect but provide a solid foundation for further exploration and refinement.

- `utils`: This directory is the backbone of the training function development, housing Python scripts that handle various essential aspects of model training. It encompasses constants, distributions, logging, optimizer and scheduler scripts, and other basic functions, such as those for evaluating metrics and the model itself.

- `debug_notebook.ipynb`: This Jupyter notebook is a powerful tool for debugging the Transformer architecture. It enables visualization of various elements, including weights, positional encodings, learning rate decay, and gradients, thereby facilitating a deeper understanding and further improvement of the model.

- `transformer_v1.py`: This Python script contains the first version of the Transformer implementation. Developed by following select tutorials, it includes two additional models: EmotionalAnalysisModel and MultilabelSequenceClassification.

- `transformer_v2.py`: This Python script hosts the second version of the Transformer, developed strictly following the original "Attention is All You Need" paper. It introduces positional encoding and implements weight retrieval, thus adhering more closely to the paper's guidelines.

## Contributing

Contributions are to this project! To contribute:

1. Fork this repository.
2. Create your feature branch: `git checkout -b feature/your-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-new-feature`
5. Submit a pull request to the `main` branch.

Please ensure your code adheres to the existing style for consistency. 

## License

This project is licensed under the MIT License, reflecting our commitment to open and accessible knowledge sharing.

## Contact Information

For any questions or feedback, please feel free to reach out to me:
- Email: vascoreid@gmail.com
- LinkedIn: [Vasco Reid](https://www.linkedin.com/in/vasco-reid-796247140/)
- GitHub: [reiid00](https://github.com/reiid00)
