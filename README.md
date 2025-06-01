## Prerequisite steps to run this repo:

1. [Create an AWS account](console.aws.amazon.com)
2. [Make an S3 bucket](https://s3.console.aws.amazon.com/s3/)
3. Put the bucket name in the .env file
4. Put the bucket's REGION in .env
5. Create an IAM User:

### 🔐 Step-by-Step: Create a New IAM User with S3 Access

1. **Sign into the AWS Management Console**  
   Go to: [https://console.aws.amazon.com/iam/](https://console.aws.amazon.com/iam/)

2. **Go to Users**  
   In the left sidebar, click **Users**, then click **“Add users”**.

3. **Set Username & Access Type**
   - **User name**: something like `metaflow-user`
   - **Access type**: ✅ **Programmatic access**  
     (this gives you an access key & secret key for use in code)

4. **Set Permissions**
   - Choose **“Attach policies directly”**
   - To keep it simple, search for and check ✅ **AmazonS3FullAccess**  
     (You can restrict this later to specific buckets if needed.)

5. **Skip tags** (optional)

6. **Review & Create**
   - Click **Create user**

7. **Save the credentials!**
   - On the success screen, you’ll see:
     - **Access key ID**
     - **Secret access key**
   - 📥 **Download the CSV** or copy them somewhere safe — you won’t see the secret again later.

8. **Put the access key ID and secret access key into the appropriate env vars in the .env file.**

## Okay, time to run the repo!

## 🚀 How to Build, Train, and Launch the Dashboard

### 🛠 Step 1: Run the setup script

From the root of the project:

```bash
./setup_and_build.sh     # for macOS/Linux
```

or for windows:

```bash
./setup_and_build.ps1    # for Windows
```

This will:
   - Download and cache the IMDB dataset locally
   - Build the Docker images for all components

### 🧱 Step 2: Start all services

From the root of the project:

```bash
docker compose up -d
```

This runs:
   - metaflow-pipeline: a container for running the ML pipeline
   - streamlit-dashboard: the web app at http://localhost:8501

### 🧪 Step 3: Train the model

Open a shell inside the Metaflow container:

```
docker exec -it metaflow-pipeline bash
```

Then run the Metaflow flow:

```
python sentiment_analysis_flow.py run --s3_bucket $S3_BUCKET
```

✅ Make sure your .env file contains a valid S3 bucket name:

```
S3_BUCKET=your-s3-bucket-name
```

This command will:
   - Fetch and preprocess book reviews
   - Train a logistic regression model
   - Evaluate the model on validation data
   - Upload metrics, model, vectorizer, and confusion matrix to S3

### 📊 Step 4: View the dashboard

Open your browser and visit:

```
http://localhost:8501
```

You’ll see:
   - 📊 Evaluation metrics (accuracy, f1_score)
   - 📉 Confusion matrix image
   - ✍️ Text box for testing new reviews
   - 🔮 Sentiment prediction for input

### 🧹 Optional: Stop all containers

```
docker compose down
```


## Commands you should not need, but I'm documenting just in case:

To download the imdb data (which is done for you already in the pushed version of this repository):
`python -c "from datasets import load_dataset; load_dataset('imdb', cache_dir='./hf_cache')"`
