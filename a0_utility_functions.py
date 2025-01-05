from typing import Dict
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve, auc,ConfusionMatrixDisplay,classification_report,recall_score,f1_score,precision_score
import seaborn as sns
from scipy.stats import ks_2samp

############################################################
###         1. Functions to Ingest Raw Data Files        ###
############################################################

def is_integer(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False

def ingest_files(directory: str) -> Dict[str, pd.DataFrame]:
    '''
    This function will ingest every CSV and text file in the specified directory
    into a pandas DataFrame. It will return a dictionary containing
    these DataFrames, keyed by the file name.
    
    It handles zipped CSV files, plain text files, and JSON data within text files.
    '''
    
    # If the directory has no trailing slash, add one
    if not directory.endswith("/"):
        directory += "/"
    
    all_files = os.listdir(directory)
    output = {}
    
    print(f"Directory {directory} has {len(all_files)} files:")
    for file_name in all_files:
        file_path = directory + file_name
        print(f"    Reading file {file_name}")
        data_dicts = []
        
        try:
            if file_name.endswith(".csv") or file_name.endswith(".zip"):
                df = pd.read_csv(file_path, dtype=str, skiprows=1)
            elif file_name.endswith(".txt"):
                with open(file_path, 'r') as file:
                    first_line = file.readline().strip()
                    # if first_line.startswith("{") and first_line.endswith("}"):  # JSON check
                    if first_line.startswith("{"):  # JSON check
                        # data_dicts = [json.loads(first_line)]
                        for line in file:
                            data_dict = json.loads(line.strip())
                            data_dicts.append(data_dict)
                        df = pd.DataFrame(data_dicts)
                    else:
                        file.seek(0)  # Reset file pointer to the beginning
                        data = file.readlines()[1:]
                        df = pd.DataFrame(data, columns=["text"])
            else:
                print(f"    Skipping unsupported file type: {file_name}")
                continue

            # Clean the data by removing rows with non-integer IDs (if applicable)
            if 'id' in df.columns:
                invalid_rows = df['id'].apply(lambda x: not is_integer(x))
                if invalid_rows.sum() > 0:
                    print(f"        Found {invalid_rows.sum()} invalid rows which were removed")
                    df = df[~invalid_rows]
            
            output[file_name] = df
        except Exception as e:
            print(f"    Failed to read file {file_name}: {e}")
    
    return output

        

def print_freq_target_rate(df, columns, target='isFraud'):
    '''
    This function print out the transaction frequencies and the target rate for each features.
    '''
    for column in columns:
        value_counts = df[column].value_counts()
        total_count = value_counts.sum()
        proportions = value_counts / total_count * 100

        # Calculate fraud counts and rates
        fraud_counts = df.groupby(column)[target].sum()
        fraud_rates = fraud_counts / value_counts * 100
        
        # Create a DataFrame for the Frequency and Mix table
        freq_mix_table = pd.DataFrame({
            'Frequency': value_counts.map('{:,}'.format), 
            'Proportion%': proportions.round(2),
            'Fraud_Count': fraud_counts.map('{:,}'.format),
            'Fraud_Rate%': fraud_rates.round(2)
        }).sort_values(by='Fraud_Rate%', ascending=False)
        
        # Display the results
        print(f"Frequency, Proportion, Fraud Counts, and Fraud Rates Table for '{column}':")
        print(freq_mix_table)
        print()


############################################################
###      2. Functions for Exploratory Data Analysis      ###
############################################################

def plot_amount_target_rate(df: pd.DataFrame, columns: list, target='isFraud', bins=20, min_amount=0):
    '''
    This function creates a histogram of transaction amounts and overlays the target rate for each bin.
    '''
    
    if len(columns) != 1:
        raise ValueError("The 'columns' list should contain exactly one column name for transaction amounts.")
    
    transaction_col = columns[0]

    # Filter the dataframe for transaction amounts greater than the specified minimum amount
    df_filtered = df[df[transaction_col] > min_amount]

    # Create bins for the histogram
    hist, bin_edges = pd.cut(df_filtered[transaction_col], bins=bins, retbins=True)

    # Calculate the fraud rate for each bin
    bin_counts = df_filtered.groupby(hist)[target].count()
    bin_fraud_counts = df_filtered.groupby(hist)[target].sum()
    fraud_rate = bin_fraud_counts / bin_counts * 100

    # Plot the histogram of transaction amounts
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Histogram plot
    ax1.hist(df_filtered[transaction_col], bins=bin_edges, color='skyblue', alpha=0.7, label='Transaction Amounts')
    ax1.set_xlabel('Transaction Amount', fontsize=8)
    ax1.set_ylabel('Frequency', fontsize=8)
    ax1.set_title('Histogram of Transaction Amounts with Fraud Rate', fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)

    # Create a secondary y-axis for fraud rate
    ax2 = ax1.twinx()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax2.plot(bin_centers, fraud_rate, color='orange', marker='o', linestyle='-', label='Fraud Rate (%)')
    ax2.set_ylabel('Fraud Rate (%)', color='orange', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=8)

    # Add legends
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, fontsize=8)

    # Show the plot
    plt.show()


def plot_freq_target_rate_multi(df, columns, target='isFraud'):
    '''
    This function creates a histogram of transaction counts and overlays the target rate for the mutiple features.
    '''
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        value_counts = df[column].value_counts()
        total_count = value_counts.sum()
        proportions = value_counts / total_count * 100

        # Calculate fraud counts and rates
        fraud_counts = df.groupby(column)[target].sum()
        fraud_rates = fraud_counts / value_counts * 100
        
        # Create a DataFrame for the Frequency and Mix table
        freq_mix_table = pd.DataFrame({
            'Frequency': value_counts,
            'Proportion%': proportions.round(2),
            'Fraud_Count': fraud_counts,
            'Fraud_Rate%': fraud_rates.round(2)
        }).sort_values(by='Fraud_Rate%', ascending=False)
        
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        freq_mix_table[['Frequency']].plot(kind='bar', ax=ax1, position=1, color='skyblue', width=0.4, legend=False)
        freq_mix_table[['Fraud_Rate%']].plot(kind='line', ax=ax2, color='orange', marker='o', linewidth=2, markersize=6, legend=False)

        ax1.set_xlabel(f"{column}", fontsize=14)
        ax1.set_ylabel('Frequency', color='skyblue', fontsize=14)
        ax2.set_ylabel('Fraud Rate (%)', color='orange', fontsize=14)
        ax1.set_title(f"{column}: Frequency and Fraud Rate", fontsize=14)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def plot_freq_target_rate(df, columns, target='isFraud'):
    '''
    This function creates a histogram of transaction counts and overlays the target rate for the mutiple features.
    '''
    for column in columns:
        value_counts = df[column].value_counts()
        total_count = value_counts.sum()
        proportions = value_counts / total_count * 100

        # Calculate fraud counts and rates
        fraud_counts = df.groupby(column)[target].sum()
        fraud_rates = fraud_counts / value_counts * 100
        
        # Create a DataFrame for the Frequency and Mix table
        freq_mix_table = pd.DataFrame({
            'Frequency': value_counts,
            'Proportion%': proportions.round(2),
            'Fraud_Count': fraud_counts,
            'Fraud_Rate%': fraud_rates.round                                            (2)
        }).sort_values(by='Fraud_Rate%', ascending=False)
        
        # Plot the Frequency and Fraud Rate
        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax2 = ax1.twinx()
        freq_mix_table[['Frequency']].plot(kind='bar', ax=ax1, position=1, color='skyblue', width=0.4)
        freq_mix_table[['Fraud_Rate%']].plot(kind='line', ax=ax2, color='orange', marker='o', linewidth=2, markersize=6)
        # ax1.bar(freq_mix_table.index, freq_mix_table['Frequency'], color='skyblue', width=0.4)
        # ax2.plot(freq_mix_table.index, freq_mix_table['Fraud_Rate%'], color='orange', marker='o', linewidth=2, markersize=6)

        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency', color='skyblue')
        ax2.set_ylabel('Fraud Rate (%)', color='orange')
        ax1.set_title(f"{column}: Frequency and Fraud Rate")
        ax1.legend(['Frequency'], loc='upper left')
        ax2.legend(['Fraud Rate (%)'], loc='upper right')
        plt.show()


def plot_freq_target_rate2(df, columns, target='isFraud'):
    '''
    This function creates a histogram of transaction counts and overlays the target rate for the mutiple features.
     - Updated version to correct some chart errors
    '''
    for column in columns:
        value_counts = df[column].value_counts()
        total_count = value_counts.sum()
        proportions = value_counts / total_count * 100

        # Calculate fraud counts and rates
        fraud_counts = df.groupby(column)[target].sum()
        fraud_rates = fraud_counts / value_counts * 100
        
        # Create a DataFrame for the Frequency and Mix table
        freq_mix_table = pd.DataFrame({
            'Frequency': value_counts,
            'Proportion%': proportions.round(2),
            'Fraud_Count': fraud_counts,
            'Fraud_Rate%': fraud_rates.round                                            (2)
        }).sort_values(by='Fraud_Rate%', ascending=False)
        
        # Plot the Frequency and Fraud Rate
        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax2 = ax1.twinx()
        # freq_mix_table[['Frequency']].plot(kind='bar', ax=ax1, position=1, color='skyblue', width=0.4)
        # freq_mix_table[['Fraud_Rate%']].plot(kind='line', ax=ax2, color='orange', marker='o', linewidth=2, markersize=6)
        ax1.bar(freq_mix_table.index, freq_mix_table['Frequency'], color='skyblue', width=0.4)
        ax2.plot(freq_mix_table.index, freq_mix_table['Fraud_Rate%'], color='orange', marker='o', linewidth=2, markersize=6)

        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency', color='skyblue')
        ax2.set_ylabel('Fraud Rate (%)', color='orange')
        ax1.set_title(f"{column}: Frequency and Fraud Rate")
        ax1.legend(['Frequency'], loc='upper left')
        ax2.legend(['Fraud Rate (%)'], loc='upper right')
        plt.show()



def plot_amount_target_rate_multi(df: pd.DataFrame, columns: list, target='isFraud', bins=20):
    '''
    This function creates a histogram of transaction amounts and overlays the target rate for the mutiple features by bins.
    '''
    
    num_plots = len(columns)
    num_cols = 2
    num_rows = (num_plots + 1) // num_cols
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()
    
    for i, transaction_col in enumerate(columns):
        # Create bins for the histogram
        hist, bin_edges = pd.cut(df[transaction_col], bins=bins, retbins=True)

        # Calculate the fraud rate for each bin
        bin_counts = df.groupby(hist)[target].count()
        bin_fraud_counts = df.groupby(hist)[target].sum()
        fraud_rate = bin_fraud_counts / bin_counts * 100

        # Plot the histogram of transaction amounts
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        # Histogram plot
        ax1.hist(df[transaction_col], bins=bin_edges, color='skyblue', alpha=0.7, label=f"{transaction_col}")
        ax1.set_xlabel(f"{transaction_col}", fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Histogram of {transaction_col} with Fraud Rate', fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Create a secondary y-axis for fraud rate
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax2.plot(bin_centers, fraud_rate, color='orange', marker='o', linestyle='-', label='Fraud Rate (%)')
        ax2.set_ylabel('Fraud Rate (%)', color='orange', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='orange', labelsize=12)

        # Add legends
        # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, fontsize=10)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=12)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



def plot_monthly_transaction_stats(df, dateTime):
    """
    Plots monthly transaction counts and fraud rates from a DataFrame.
    """
    # Ensure the dateTime column is in datetime format
    df[dateTime] = pd.to_datetime(df[dateTime])

    # Extract year and month from dateTime
    df['year_month'] = df[dateTime].dt.to_period('M')

    # Group by year_month and calculate counts and fraud rates
    monthly_stats = df.groupby('year_month').agg(
        transaction_count=('isFraud', 'size'),
        fraud_count=('isFraud', 'sum')
    ).reset_index()
    monthly_stats['fraud_rate'] = monthly_stats['fraud_count'] / monthly_stats['transaction_count'] * 100

    # Convert year_month back to datetime for plotting
    monthly_stats['year_month'] = monthly_stats['year_month'].dt.to_timestamp()

    # Display the results
    print("Monthly Transaction Stats:")
    print(monthly_stats)

    # Plot the monthly transaction counts and fraud rates
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plot transaction counts as bars
    ax1.set_xlabel('Month', fontsize=8)
    ax1.set_ylabel('Transaction Count', color='tab:blue', fontsize=8)
    ax1.bar(monthly_stats['year_month'], monthly_stats['transaction_count'], color='skyblue', alpha=0.6, label='Transaction Count', width=15, edgecolor='skyblue', linewidth=0)
    ax1.tick_params(axis='x', labelsize=8)  # Set x-axis label size
    ax1.tick_params(axis='y', labelcolor='skyblue', labelsize=8)
    ax1.set_ylim(0, 70000)  # Set y-axis limit to 70,000

    # Create a second y-axis for fraud rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fraud Rate (%)', color='orange', fontsize=8)
    ax2.plot(monthly_stats['year_month'], monthly_stats['fraud_rate'], color='orange', marker='o', linestyle='-', label='Fraud Rate (%)')
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=8)

    # Add titles and labels
    plt.title('Monthly Transaction Counts and Fraud Rates', fontsize=8)

    # Add legends
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9), fontsize=8)

    # Show plot
    fig.tight_layout()
    plt.show()


############################################################
###    3. Functions for Model Building and Evaluation    ###
############################################################

def evaluate_model(y_test, y_pred):
    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred)
    print("Test AUC:", auc)
    
    # Print classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', colorbar=False)
    disp.ax_.set_title("Confusion Matrix")
    plt.show()
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # the most features that had an impact of our classes
def plot_feature_importance(model, feature_names=None, plot=True):

    feature_importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = model.feature_name()

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)#.head(top_n)

    if plot:
        plt.figure(figsize=(10, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.show()

    return feature_importance_df


def dict_eval(y_true, y_pred, model_name):

    ks_statistics, p_value = ks_2samp(y_pred[y_true == 0], y_pred[y_true == 1])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    eval = {'Model':model_name,
            'Recall':recall_score(y_true, y_pred),
            'FPR':fpr[1],
            'AUC':roc_auc_score(y_true, y_pred),
            'ACC':accuracy_score(y_true, y_pred),
            'precision':precision_score(y_true, y_pred),
            'F1':f1_score(y_true, y_pred),
            'KS':ks_statistics}
    return eval

