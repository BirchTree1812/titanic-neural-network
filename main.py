import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import neural_network as neural
import sys
import time as t
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# allows the user to specify the path of the file that they want to use
datapath = sys.argv[1]
# allows the user to choose the exact part of code they want to run
choice = sys.argv[2]

# Open from the CSV file
df = pd.read_csv(datapath)



# Map sex and embark values to numbers
sex_mapping = {'male': 0, 'female': 1}
embark_mapping = {'S': 0, 'C': 1, 'Q': 2}
df['Sex'] = df['Sex'].map(sex_mapping)
df['Embarked'] = df['Embarked'].map(embark_mapping)

# Determine medians for sex and embark, once they're mapped to numbers
age_median = df['Age'].median()
fare_median = df['Fare'].median()
embark_median = df['Embarked'].median()

# Fill the missing values of sex and embark with the medians
df = df.fillna({'Age': age_median,'Fare': fare_median, 'Embarked': embark_median})

# Drop columns that are not useful for correlation
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# if the user writes 1 in the command line, print the first 20 rows of the dataframe after the preprocessing
if choice == "1":
    # Print once again
    print(df.head(20).to_string(index=False))


# if the user writes 2 in the command line, this will create a correlation matrix and 
# identify variables that are least-correlated to survival
if choice == "2":
    # this checks how much time it takes to run the code
    start = t.time()

    # create the correlation matrix
    correlation_matrix = df.corr()
    df.corr()

    # Visualize the correlation matrix
    # Set your threshold here
    threshold = 0.3

    # Mask for values below the threshold
    mask = np.abs(correlation_matrix) < threshold

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap with Threshold')
    plt.savefig("correlation_matrix.webp")
    plt.close()
    # Extract correlation values for 'Survived'
    survived_corr = correlation_matrix['Survived'].drop('Survived')

    # Identify variables with the least correlation to 'Survived'
    least_corr_variables = survived_corr.abs().nsmallest(10)

    # Display the least correlated variables
    print("Variables with the least correlation to 'Survived':")
    print(least_corr_variables)

    end = t.time()
    # show how much time it took to run the code
    print("Time taken to run code:", end-start, "secs")

# if the user write 3 in the command line, this will train the neural network
if choice == "3":
    start = t.time()
    # Split the dataset into the target variable and the features
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # split the dataset into the training and testing parts. This means both the target variable and the features get split that way
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ready to preprocess the data
    categorical_features = ['Pclass','Embarked']
    binary_features = ['Sex']
    continuous_features = [ 'PassengerId', 'SibSp', 'Age', 'Parch', 'Fare']



    # plotting diagrams
    # histograms of continuous features
    fig, axs = df[['SibSp', 'Age', 'Parch', 'Fare']].hist(bins=30, figsize=(10, 10))
    plt.suptitle('Distribution of Continuous Features')
    plt.subplot(2, 2, 1)
    plt.title("Parents and children")
    plt.subplot(2, 2, 3)
    plt.title("Siblings and spouses")
    plt.savefig("continuous_features_distribution.webp")
    plt.close()

    # bar plots of categorical features. Plots several subplots
    plt.figure(figsize=(10,25))
    # index of the current subplot
    index = 1
    binary_categorical_amount = len(categorical_features+binary_features)+1
    for feature in categorical_features+binary_features:
        plt.subplot(binary_categorical_amount, 1, index)
        sns.countplot(x=feature, data=df)
        plt.title(f'Distribution of {feature}')
        index+=1
    plt.savefig("categorial_binary_feature_distribution.webp")
    plt.close()


    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('bin', 'passthrough', binary_features),
            ('cont', StandardScaler(), continuous_features)
        ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)






    # Initializing the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = neural.NeuralNetwork(input_dim)
    # BCELoss is good for binary classification
    criterion = nn.MSELoss() 
    # this optimizer has weight decay specified, which allows it to adjust learning rate as it goes. It should reduce overfitting.
    optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-5)
    # Convert data to PyTorch tensors. A necessary step.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


    # This line of code toggles debug-related functions. If debugging is necessary, I set it to True. Otherwise, I set it to False. 
    debug = False



    # this code is a mechanism, that stops earlier than usual, if there's no improvement for too long
    # this represents the best loss value.
    best_test_loss = float('inf')
    # early stopping patience is a threshold for number of epochs without improvement. If it exceeds the threshold, the training is stopped
    early_stopping_patience = 10
    # no_improvement_counter determines for how many epochs there has been no improvement in loss
    no_improvement_counter = 0

    # Training loop. Run through each epoch, training the model.
    num_epochs = 100
    train_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    for epoch in range(num_epochs):    

        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_loss_list.append(total_train_loss)
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss:.4f}')



        # Test
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0.0
        
        with torch.no_grad():
            predictions = model(X_test_tensor)
            test_loss = criterion(predictions, y_test_tensor)
            test_predictions = (predictions > 0.5).float()
            test_accuracy = accuracy_score(y_test_tensor, test_predictions)
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)
        
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        if no_improvement_counter >= early_stopping_patience:
            print("Early stopping triggered")
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {best_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            break
        # these lines of code are useful for debugging, to see if there are any mistakes in the way the tensors are processed. Turned off this time
        if (epoch+1) % 10 == 0 and debug == True:
            if torch.isnan(outputs).any():
                print("NaNs detected in outputs")
            print("Input:", X_train_tensor[:5])
            print("Output:", outputs[:5])
    

    # plotting a few more graphs for the training process
    # loss curve
    plt.plot(range(epoch+1), train_loss_list, label='Training Loss')
    plt.plot(range(epoch+1), test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig("loss_curve.webp")
    plt.close()

    # accuracy curve
    plt.plot(range(epoch+1), test_accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Curve')
    plt.savefig("test_accuracy_curve.webp")
    plt.close()

    # save the neural network. I know that it's possible to save only the state dictionary, 
    # but I thought that downloading the netire neural netowrk is more trustworthy
    torch.save(model, 'titanic_model.pth')

    end  = t.time()
    print("Time taken to execute is", end-start, "secs")
