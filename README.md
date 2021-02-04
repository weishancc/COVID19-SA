# COVID19-SA
Covid-19 sentiment analysis by fine-tuning bert model basesd on social network.
<br/> Also, this repo provides and e2e demo 👨‍💻
<p align="left"><img width=20% src="https://github.com/weishancc/COVID19-SA/blob/master/images/header.PNG"></p>

## 📝 Requirements
* [Flask](https://pypi.org/project/Flask/)
* [Pytorch](https://pypi.org/project/torch/)
* [Transformers](https://pypi.org/project/transformers/)

## 📔 Usage
### Fine-tune bert model
[Optional] Create and activate your virtual environments first!
```console
$ conda create -n venv
$ source activate venv
```
Begin to fine-tune
```console
$ python main.py
```
When training complete, [main.py](https://github.com/weishancc/COVID19-SA/blob/master/main.py) will save accuracy and loss history(training/validate), then we provide two functions in [predict.py](https://github.com/weishancc/COVID19-SA/blob/master/predict.py) to get inference (**get_predictions** / **get_prediction_with_single**). This step will infernece from test dataloader using function ```get_predictions```, where live-demo will inference from web's input in live-demo using function ```get_prediction_with_single```.

### Live-demo
We use flask as web server, and interact with web client via ajax. Due to file size limit, we place fine-tuned model weight [here](https://drive.google.com/file/d/1G72u37FknxwUGKaEXo_rWiM8Ib6QNLFv/view), of course, you can change to your weight under folder [web_demo](https://github.com/weishancc/COVID19-SA/blob/master/web_demo)
<p align="left"><img width=50% src="https://github.com/weishancc/COVID19-SA/blob/master/images/architecture.PNG"></p>

#### Start web server
```console
$ cd web_demo
$ python web.py
```
#### Then you will see the website on localhost:5000 :)
We would like to add [bertviz](https://github.com/jessevig/bertviz) at first, but it seems unsupport on html for now.
<p align="left"><img width=80% src="https://github.com/weishancc/COVID19-SA/blob/master/images/screenshot.PNG"></p>

## 📔 Reference
### Dateset
* [https://www.kaggle.com/surajkum1198/datasets](https://www.kaggle.com/surajkum1198/datasets)

### Implementation
* [https://www.kaggle.com/alankritamishra/covid-19-tweet-sentiment-analysis](https://www.kaggle.com/alankritamishra/covid-19-tweet-sentiment-analysis)
* [https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html#1.-%E6%BA%96%E5%82%99%E5%8E%9F%E5%A7%8B%E6%96%87%E6%9C%AC%E6%95%B8%E6%93%9A](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html#1.-%E6%BA%96%E5%82%99%E5%8E%9F%E5%A7%8B%E6%96%87%E6%9C%AC%E6%95%B8%E6%93%9A)
