# ---- Imports: bring in tools we need ----

import requests   # lets Python send messages over the network (to Ollama)
import time       # lets us pause or measure time if needed
import pickle     # lets us save Python data to a file (for checkpoints)
import os         # lets us work with files and folders on the computer
import pandas as pd  # lets us read and save Excel files
import queue         # used by the threaded generation helper
import threading     # used to run generation in a separate thread


# ---- Settings: where Ollama lives and which model to use ----

# Ollama runs on the server and listens at this address.
# "localhost" means "this same machine", 11434 is Ollama's door number (port).
OLLAMA_URL = "http://localhost:11434/api/generate"

# The name of the model you pulled with "ollama pull mistral"
MODEL = "mistral"


# ---- The function that sends a prompt and gets an answer ----

def generate_with_timeout(prompt, timeout_seconds=120):
    # "def" means we are defining a reusable block of code (a function).
    # It takes two inputs:
    #   prompt          = the text you want the model to respond to
    #   timeout_seconds = how long to wait before giving up (default 120s)

    try:
        # "try" means: attempt the code below, and if anything
        # goes wrong, jump to the matching "except" instead of crashing.
        # Send the prompt to Ollama and wait for the reply.
        r = requests.post(
            OLLAMA_URL,            # where to send it
            json={                 # the data we send, as a package:
                "model": MODEL,        # which model to use
                "prompt": prompt,      # your actual question/text
                "stream": False,       # False = give me the whole answer at once
                "options": {
                    "num_predict": 512 # max length of the answer (in tokens)
                }
            },
            timeout=timeout_seconds  # give up if it takes longer than this
        )

        # If Ollama replied with an error code (like 404 or 500),
        # this line turns it into a Python error so "except" catches it.
        r.raise_for_status()

        # The reply comes back as JSON (structured text).
        # .json() unpacks it, and ["response"] pulls out the answer text.
        answer = r.json()["response"]

        # Give back two things: the answer, and the word "ok" as a status.
        return answer, "success"

    except requests.Timeout:
        # We land here if the model took too long.
        # Return no answer (None) and the status "timeout".
        return None, "timeout"

    except Exception as e:
        # We land here for any other problem (server down, wrong model name...).
        # "e" holds the error message, which we include in the status.
        return None, f"error: {e}"


# ---- Example of using the function ----

# Call the function with a test prompt and store the two results.
answer, status = generate_with_timeout("Say hello in Swedish.")

# Print what happened so you can see it in the terminal.
print("Status:", status)
print("Answer:", answer)

def analyze_author_topic_pairing(comment1, comment2, topic1, topic2, timeout_seconds=1200):
    """
    Analyzes whether two comments are from the same author and whether they discuss the same topic.
    Returns a tuple of (analysis_result, status).
    """
    # Create a prompt for the model to assess author and topic consistency
    prompt = f"""Analyze the following two comments and determine:
1. Are these comments likely written by the same author? (Consider writing style, vocabulary, tone)
2. Do these comments discuss the same topic or subreddit?

Comment 1 (Topic: {topic1}): {comment1}

Comment 2 (Topic: {topic2}): {comment2}

Respond with:
- Same Author: Yes/No
- Same Topic: Yes/No
- Confidence (High/Medium/Low)
- Brief explanation"""

    return generate_with_timeout(prompt, timeout_seconds)


def load_xlsx(file_path):
    """
    Load an Excel file and return as a pandas DataFrame.
    """
    print(f"Attempting to load xlsx file from: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    df = pd.read_excel(file_path)
    return df


# ============= CHECKPOINT FUNCTIONS =============

def save_checkpoint(checkpoint_path, results, skipped_rows, successful_rows, last_processed_index):
    """
    Save progress to a checkpoint file.
    """
    checkpoint_data = {
        'results': results,
        'skipped_rows': skipped_rows,
        'successful_rows': successful_rows,
        'last_processed_index': last_processed_index
    }
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"✓ Checkpoint saved at row {last_processed_index + 1}")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path):
    """
    Load progress from a checkpoint file.
    Returns (results, skipped_rows, successful_rows, last_processed_index) or None if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"✓ Checkpoint loaded! Resuming from row {checkpoint_data['last_processed_index'] + 2}")
        return (checkpoint_data['results'],
                checkpoint_data['skipped_rows'],
                checkpoint_data['successful_rows'],
                checkpoint_data['last_processed_index'])
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # ============= BATCH PROCESSING CONFIGURATION =============
    # Set ONE of these batch limits (the script will stop when either limit is reached)

    BATCH_ROW_LIMIT = None  # Stop after processing X rows (set to None to disable)
    BATCH_TIME_LIMIT_MINUTES = None  # Stop after X minutes (set to None to disable)

    # Convert time limit to seconds
    BATCH_TIME_LIMIT_SECONDS = BATCH_TIME_LIMIT_MINUTES * 60 if BATCH_TIME_LIMIT_MINUTES else None

    CHECKPOINT_SAVE_FREQUENCY = 5  # Save checkpoint every 5 rows

    # Load Excel file using relative path
    file_path = os.path.join("/home/dsvex26g2/mistral_env/dast_1000.xlsx")
    df = load_xlsx(file_path)

    if df is not None:
        # Create column aliases for privacy and clarity
        # Columns: A-B are usernames, C-D are topics/subreddits, E-F are comments
        column_mapping = {
            'A': 'username_1',
            'B': 'username_2',
            'C': 'topic_1',
            'D': 'topic_2',
            'E': 'comment_1',
            'F': 'comment_2'
        }

        # Rename columns based on mapping
        df = df.rename(columns=dict(zip(df.columns, column_mapping.values())))

        print(f"Loaded {len(df)} rows for analysis")
        print("---")
        print(f"BATCH CONFIGURATION:")
        print(f"  Row limit: {BATCH_ROW_LIMIT if BATCH_ROW_LIMIT else 'Unlimited'}")
        print(f"  Time limit: {BATCH_TIME_LIMIT_MINUTES} minutes" if BATCH_TIME_LIMIT_MINUTES else "  Time limit: Unlimited")
        print("---\n")

        # ============= CHECKPOINT LOADING =============
        checkpoint_path = "/home/dsvex26g2/mistral_env/dast_1000_checkpoint.pkl"
        checkpoint_info = load_checkpoint(checkpoint_path)

        if checkpoint_info:
            results, skipped_rows, successful_rows, last_processed_index = checkpoint_info
            start_index = last_processed_index + 1
        else:
            results = []
            skipped_rows = []
            successful_rows = 0
            start_index = 0

        print(f"Starting from row {start_index + 1}/{len(df)}")
        print("---\n")

        # ============= BATCH PROCESSING LOOP =============
        batch_start_time = time.time()
        rows_processed_in_batch = 0
        batch_complete = False

        for index in range(start_index, len(df)):
            # ============= CHECK BATCH LIMITS =============
            # Check row limit
            if BATCH_ROW_LIMIT and rows_processed_in_batch >= BATCH_ROW_LIMIT:
                print(f"\n⏸️  BATCH LIMIT REACHED: {BATCH_ROW_LIMIT} rows processed")
                batch_complete = True
                break

            # Check time limit
            if BATCH_TIME_LIMIT_SECONDS:
                elapsed_time = time.time() - batch_start_time
                if elapsed_time >= BATCH_TIME_LIMIT_SECONDS:
                    print(f"\n⏸️  TIME LIMIT REACHED: {BATCH_TIME_LIMIT_MINUTES} minutes elapsed")
                    batch_complete = True
                    break

            row = df.iloc[index]
            username_1 = row['username_1']
            username_2 = row['username_2']
            topic_1 = row['topic_1']
            topic_2 = row['topic_2']
            comment_1 = row['comment_1']
            comment_2 = row['comment_2']

            # ============= VALIDATE DATA =============
            # Check for missing or null values
            if pd.isna(comment_1) or pd.isna(comment_2) or pd.isna(username_1) or pd.isna(username_2):
                print(f"Row {index + 1}:")
                print(f"⚠️  SKIPPED - Row contains missing data (null/NaN values)")
                skipped_rows.append({
                    'row': index + 1,
                    'reason': 'missing_data'
                })
                results.append({
                    'row': index + 1,
                    'status': 'skipped_missing_data',
                    'analysis': '[SKIPPED - Missing Data]'
                })
                save_checkpoint(checkpoint_path, results, skipped_rows, successful_rows, index)
                rows_processed_in_batch += 1
                print("---\n")
                continue  # Skip this row and move to the next one

            # Convert to strings to handle any edge cases
            comment_1 = str(comment_1)[:1000]
            comment_2 = str(comment_2)[:1000]
            username_1 = str(username_1)
            username_2 = str(username_2)

            print(f"Row {index + 1}:")
            print(f"Author 1 (anon): {username_1[:3]}***")
            print(f"Author 2 (anon): {username_2[:3]}***")
            print(f"Topic 1: {topic_1}")
            print(f"Topic 2: {topic_2}")
            print(f"Comment 1: {comment_1[:100]}...")
            print(f"Comment 2: {comment_2[:100]}...")

            # Analyze the pairing with timeout handling
            analysis_result, status = analyze_author_topic_pairing(
                comment_1, comment_2, topic_1, topic_2, timeout_seconds=1200
            )

            if status == "success":
                print(f"Analysis Result:")
                print(analysis_result)
                successful_rows += 1
                results.append({
                    'row': index + 1,
                    'status': 'completed',
                    'analysis': analysis_result
                })
            else:
                print(f"⚠️  SKIPPED - Status: {status}")
                skipped_rows.append({
                    'row': index + 1,
                    'reason': status
                })
                results.append({
                    'row': index + 1,
                    'status': status,
                    'analysis': '[SKIPPED]'
                })

            print("---")

            rows_processed_in_batch += 1

            # ============= SAVE CHECKPOINT EVERY FEW ROWS =============
            if rows_processed_in_batch % CHECKPOINT_SAVE_FREQUENCY == 0:
                save_checkpoint(checkpoint_path, results, skipped_rows, successful_rows, index)

        # Ensure checkpoint is saved at end of batch
        save_checkpoint(checkpoint_path, results, skipped_rows, successful_rows, index)

        # ============= PRINT BATCH SUMMARY =============
        print("\n" + "="*50)
        if batch_complete:
            print(f"BATCH PROCESSING PAUSED")
            print(f"Rows processed in this batch: {rows_processed_in_batch}")
            print(f"Total rows processed overall: {index + 1}/{len(df)}")
            print(f"Successful rows: {successful_rows}/{index + 1}")
            print(f"Skipped rows: {len(skipped_rows)}/{index + 1}")
            print(f"\nCheckpoint saved. Run the script again to continue.")
        else:
            print(f"PROCESSING COMPLETE")
            print(f"Successful rows: {successful_rows}/{len(df)}")
            print(f"Skipped rows: {len(skipped_rows)}/{len(df)}")

        if skipped_rows:
            print("\nSkipped row details:")
            for skipped in skipped_rows:
                print(f"  Row {skipped['row']}: {skipped['reason']}")

        # ============= CLEAN UP CHECKPOINT AFTER SUCCESS =============
        if not batch_complete and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"\nCheckpoint file removed after successful completion.")
        elif batch_complete and os.path.exists(checkpoint_path):
            print(f"\nCheckpoint file preserved for resumption.")

        print("="*50 + "\n")

        # Save results to Excel file
        results_df = pd.DataFrame(results)
        results_output_path = os.path.join("/home/dsvex26g2/mistral_env/dast_1000_results.xlsx")
        results_df.to_excel(results_output_path, index=False)
        print(f"Results saved to: {results_output_path}")

        # Save skipped rows log
        if skipped_rows:
            skipped_df = pd.DataFrame(skipped_rows)
            skipped_output_path = os.path.join("/home/dsvex26g2/mistral_env/dast_1000_skipped.xlsx")
            skipped_df.to_excel(skipped_output_path, index=False)
            print(f"Skipped rows log saved to: {skipped_output_path}")
    else:
        print("Failed to load file. Exiting.")
