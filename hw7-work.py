import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
import numpy as np
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import get_scheduler, RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))

class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]
        answer = [1 if answer is True or answer.lower() == "yes" else 0]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        #input_encoding = question + " [SEP] " + passage
        ## Trying without passage first 
        encoded_review = self.tokenizer.prepare_seq2seq_batch(
            [question],
            # add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        return {
            'input_ids': encoded_review['input_ids'][0],  # we only have one example in the batch
            'attention_mask': encoded_review['attention_mask'][0],
            # attention mask tells the model where tokens are padding
            'labels': torch.tensor(answer, dtype=torch.long)  # labels are the answers (yes/no)
        }

def final_accuracy(predicted_answers, correct_answers):

    total_correct = 0 
    total = len(predicted_answers)

    for predic_, correct_, in zip(predicted_answers, correct_answers):
        if predic_.lower() == correct_.lower():
            total_correct +=1 
    
    acc = total_correct /total 
    return acc
def evaluate_model(model, dataloader, device, tokenizer=None):
    """ Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval() 
    predicted_answers = []
    correct_answers = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ##get the predicted answers sequence 
        predicted_answer_seqs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        ## decode the answer using the tokenizer 
        decoded_answers =[tokenizer.decode(predic_answer[0], skip_special_tokens=True) for predic_answer in predicted_answer_seqs]
        predicted_answers.extend(decoded_answers)
        correct_answers.extend(batch['labels'])

        ## 
        # output = model(input_ids=input_ids, attention_mask=attention_mask)

        # predictions = output.logits.to(device)
        # predictions = torch.argmax(predictions, dim=1)
        # dev_accuracy.add_batch(predictions=predictions, references=batch['labels']) 
    
    ## calculate loss 

    # compute and return metrics
    return final_accuracy(decoded_answers, correct_answers)


# Note that we need tokenizer in the training function in case of RAG
def train(mymodel, num_epochs, train_dataloader, validation_dataloader, device, lr, tokenizer=None):
    """ Train a PyTorch Module
    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :param transformers.RagTokenizer tokenizer: tokenizer used to tokenize the sentences
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    loss = torch.nn.CrossEntropyLoss()
    ## store accuracies 
    train_accuracies_store = []
    eval_accuracies_store = []
    for epoch in range(num_epochs):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')
        predicted_answers = []
        correct_answers = []

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):
            # get input ids, attention masks and labels 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # get the output from the model 
            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            # get logits and calculate loss 
            predictions = output.logits.to(device)
            model_loss = loss(predictions,  labels) 
            ##get the predicted answers sequence 
            predicted_answer = mymodel.generate(input_ids=input_ids, attention_mask=attention_mask)
            ## decode the answer using the tokenizer 
            decoded_answer = tokenizer.decode(predicted_answer[0], skip_special_tokens=True)
            ## add both the answers and correct answers to the list 
            predicted_answers.append(decoded_answer)
            correct_answers.append(batch['labels'])

            model_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad() 
            train_accuracy.add_batch(predictions=predictions,  references = batch['labels'].to(device)) 
        
        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        acc_val = train_accuracy.compute()
        print(f" - Average training metrics: accuracy={acc_val}")
        ## result
        train_accuracies_store.append(acc_val)
        
        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        eval_accuracies_store.append(val_accuracy)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        train_acc_list = [d['accuracy'] for d in train_accuracies_store]
        eval_acc_list = [d['accuracy'] for d in eval_accuracies_store]
    
    plt.plot(range(1, num_epochs+1), eval_acc_list)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    # plt.show()
    plt.savefig(str(mymodel.name) + " eval " + str(lr) + " " + str(num_epochs) + '.png')
    plt.close()
    return max(eval_acc_list)



def pre_process(model_name, batch_size, device, small_subset):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        dataset_train_subset = dataset['train'][:10]
        dataset_dev_subset = dataset['train'][:10]
        dataset_test_subset = dataset['train'][:10]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        dataset_train_subset = dataset['train'][:8000]
        dataset_dev_subset = dataset['validation']
        dataset_test_subset = dataset['train'][8000:]

    print("Size of the loaded dataset:")
    print(f" - train: {len(dataset_train_subset['passage'])}")
    print(f" - dev: {len(dataset_dev_subset['passage'])}")
    print(f" - test: {len(dataset_test_subset['passage'])}")

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    ## use tokenizer for RAG 
    mytokenizer = RagTokenizer.from_pretrained(model_name)
    ## passed in model
    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset['passage']),
        questions=list(dataset_dev_subset['question']),
        answers=list(dataset_dev_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset['passage']),
        questions=list(dataset_test_subset['question']),
        answers=list(dataset_test_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...") 

    retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)
    pretrained_model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever) 
    
    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader, mytokenizer 


def experiment(model_name, train_dataloader, validation_dataloader, device):
    best_selected_acc = -np.inf
    best_selected_params = []
    best_model = None
    for epoch in [5, 7, 9]:
        for lr in [1e-4, 5e-4, 1e-3]:
            # model = T5ForConditionalGeneration.from_pretrained(model_name, num_labels = 2)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            model.to(device)
            model.name = model_name
            best_dev_acc = train(model, epoch, train_dataloader, validation_dataloader, device, lr)
            if best_dev_acc > best_selected_acc:
                print(f" -> Revising the best accuracy on dev from {best_selected_acc} to {best_dev_acc} ")
                best_selected_acc = best_dev_acc
                best_selected_params = [epoch, lr]
                torch.save(model, "best_model " + model_name + " .pth")
    
    return best_selected_params, best_selected_acc

# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"

    # load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader, tokenizer = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset)
    # pretrained_model, train_dataloader, validation_dataloader, test_dataloader = t5_pre_process(args.model,
    #                                                                                          args.batch_size,
    #                                                                                          args.device,
    #                                                                                          args.small_subset)
    print(" >>>>>>>>  Starting training ... ")
    ## added
    n_epoch = args.num_epochs
    dvice = args.device
    lr_ = args.lr

    # train(pretrained_model, n_epoch , train_dataloader, validation_dataloader, dvice, lr_)
    best_selected_params, best_selected_acc = experiment(args.model, train_dataloader, validation_dataloader, dvice)
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()
    best_model = torch.load("best_model " + args.model + " .pth")

    val_accuracy = evaluate_model(best_model, validation_dataloader, dvice, tokenizer)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(best_model, test_dataloader, dvice )
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
