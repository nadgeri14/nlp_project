# NLP Project (group 13)

### Members: Abhishek Nadgeri, Ulvi Shukurzade

## 1. Dataset overview
The dataset FACTOID: is a user-level **FAC**tuality and p**O**litical b**I**as **D**ataset, that contains a set of 4,150 news-spreading users with 3.3M Reddit posts in discussions on contemporary political topics, covering the time period from January 2020 to April 2021 on individual user level.

## 2. Setup

### 2.1 Environment Setup

* Data preprocessing support libraries

    ```pip install -r requirements.txt```

* Model library ([Pytorch geometric](https://github.com/pyg-team/pytorch_geometric#nightly-and-master))

    ```pip install pyg-nightly```

## 3. Pre-Process Data

### 3.1 Reddit Posts Crawling

   Crawl reddit posts using the ids provided in the dataset and fill the empty strings inside the dataframe.  

### 3.2 User Embeddings

  First extract user vocabularies 

  ```
  python create_vocabs_per_month.py --base_dataset=../data/reddit_dataset/factoid_dataset.gzip
  ```

  Then run the codes to generate
          
   * UBERT embeddings

  ```
   python user_embeddings_per_month.py --vocabs_dir='../data/user_vocabs_per_month' --base_dataset='../data/reddit_dataset/factoid_dataset.gzip'
  ```

### 3.3 Generate Graphs and Samples

  To generate graph samples, example script.

  ```
  python source_graph_generation.py \
  --gen_source_graphs=True \
  --path='../data/reddit_dataset/linguistic/cosine/avg/bert_embeddings/' \ 
  --base_dataset='../data/reddit_dataset/factoid_dataset.gzip' \
  --doc_embedding_file_path='../data/embeddings/bert/' \
  --embed_type='bert' \
  --merge_liwc='false' \
  --dim=768 \
  --embed_mode='avg' |& tee ../logs/graph_generation.txt
  ```

  Then after creating the graph samples, run the following to make the split

  ```
  python model_dataloader.py \
  --n_users=200 \
  --n_train_samples=1000 \
  --n_val_samples=200 \
  --base_dataset='../data/reddit_dataset/factoid_dataset.gzip' \
  --source_frames='../data/reddit_dataset/linguistic/cosine/avg/bert_embeddings/source' \
  --sample_dir='../data/reddit_dataset/model_samples_avg/bert_embeddings/' \
  --user_ids='../data/reddit_dataset/user_splits/' \
  --threshold=0.8 |& tee ../logs/model_dataloader.txt
  ```

### 3.4 Training Model

  After training, validation, test samples are created, run the model using the following. Change the parameters based on the model you want to use. The argument `gnn`  takes the following values `['gat', 'transformer']`

  ```
  python training_graph.py --patience=40 \
  --run_id='bert_embeddings' \
  --sample_dir='../data/reddit_dataset/model_samples_avg/bert_embeddings/'  \
  --result_dir='../results/' \
  --checkpoint_dir='../results/checkpoints/' \
  --max_epochs=50 \
  --learning_rate=5e-5 \
  --nheads=4 \
  --dropout=0.2 \
  --nhid_graph=256 \
  --nhid=128 \
  --users_dim=768 \
  --gnn='gat' |& tee ../logs/graph_model_main.txt
  ```
## 4. Acknowledgment
The code is built on the following work - 
* [FACTOID](https://github.com/caisa-lab/FACTOID-dataset/tree/main)
* [Temporal Graph Analysis of Misinformation Spreaders in Social Media](https://github.com/caisa-lab/textgraph22-temporal-misinformation-spreaders)