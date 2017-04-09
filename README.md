# COMPGW02/M041: the Web Economics Project

## SUMMARY

In this assignment, you are required to work on an online
advertising problem. You will help advertisers to form a
bidding strategy in order to place their ads online in a realtime
bidding system. You are required to train a bidding
strategy based on an impression training set. The aim of
this project is to help you understand some basic concepts
and write a computer program in real-time bidding based
display advertising. As you will be evaluated both as a group
as well as individually, part of the assignment is to train a
model of your choice independently. The performance of the
model trained by your team, which is either a combination
of the individually developed models or the best performing
individually developed model, will be (mainly) evaluated
on the Click-through Rate achieved on a provided test set.
In order for you to properly evaluate the performance of
each of your models before that, a benchmark click-through
rate on the validation set will be provided. Before the final
submission, you are also given the opportunity to hand in
the preliminary result of your team’s model on the test set,
which allows you to compare the performance to that of your
peers.

## DATASET

[Download](https://drive.google.com/file/d/0B73mmT9K2b4EZkZacFVBRDJtdzQ/view)

This data comes in CSV format, the first line in the file
containing the header formatted as described in Table 1.
As the testing set is used for final evaluation purposes, it
does not contain the three fields: ‘bidprice’, ‘payprice’ and
‘click’. Note that all numbers related to money (e.g., bid
price, paying price and floor price) use the currency of RMB
and the unit of Chinese fen × 1000, corresponding to the
commonly adopted cost-per-mille (CPM) pricing model.

Table 1: Fields in dataset

|Field  |Example|Supplement|
|-------|-------|-------|
|click |1 |1 if clicked, 0 if not.
|weekday |1
|hour |12
|bidid |fdfe...b8b21
|logtype |1
|userid |u_Vh1OPkFv3q5CFdR
|useragent |windows_ie
|IP |180.107.112.*
|region |80
|city |85
|adexchange |2
|domain |trq...Mi
|url |d48a...4efeb
|urlid |as3d...34frg
|slotid |2147...813
|slotwidth |300
|slotheight |250
|slotvisibility |SecondView
|slotformat |Fixed
|slotprice |0
|creative |hd2v...jhs72
|bidprice |399
|payprice |322 |Paid price afterwin the bidding.
|keypage |sasd...47hsd
|advertiser |2345
|usertag |123,5678,3456 |Contains multivalues,‘,’ as segmentation.

# DEPENDENCIES

1. [libFM](http://www.libfm.org/) by Steffen Rendle.

2. [libFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/) by Yuchin Juan.

3. Please run `pip install -r requirements.txt` to install other python dependencies.

# REFERENCES

- [Weinan Zhang, Shuai Yuan, Jun Wang, and Xuehua
Shen. Real-time bidding benchmarking with ipinyou
dataset. arXiv preprint arXiv:1407.7073, 2014.](https://arxiv.org/pdf/1407.7073.pdf)

- [Weinan Zhang, Shuai Yuan, and Jun Wang. Optimal
real-time bidding for display advertising. In Proceedings
of the 20th ACM SIGKDD international conference on
Knowledge discovery and data mining, pages 1077–1086.
ACM, 2014.](http://wnzhang.net/papers/ortb-kdd.pdf)


