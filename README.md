# Fine-grained Sentiment Analysis

## Tasks

- ABSA: Aspect Based Sentiment Analysis
- E2E-ABSA
- Complete ABSA
- ATE & OTE: Aspect Term Extraction and Opinion Term Extraction
- Emotion Analysis



| Tasks            | Input            | Output                                      |
| ---------------- | :--------------- | ------------------------------------------- |
| ABSA             | sentence, aspect | aspect sentiment                            |
| E2E-ABSA         | sentence         | aspect term, aspect sentiment               |
| Complete-ABSA    | sentence         | aspect term, aspect sentiment, opinion term |
| ATE&OTE          | sentence         | aspect term, opinion term                   |
| Emotion Analysis | sentence         | joy, anger, fear, etc.                      |
|                  |                  |                                             |

------

## Paper list

### ABSA

- **[ACL-2017]** Recurrent Attention Network on Memory for Aspect Sentiment Analysis [[paper]](https://www.aclweb.org/anthology/D17-1047/)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/Recurrent%20Attention%20Network%20on%20Memory%20for%20Aspect%20Sentiment%20Analysis/RAM%E5%9B%BE/image-01.png" style="zoom:50%;" />



- **[IJCAI-2017]** Interactive Attention Networks for Aspect-Level Sentiment Classification [[paper]](https://arxiv.org/abs/1709.00893)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/IAN%E5%9B%BE/image-01.png" style="zoom:50%;" />



- **[IJCAI-2017]** Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks [[paper]](https://arxiv.org/abs/1709.00893)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/AOA%E5%9B%BE/image-01.png" style="zoom:50%;" />





- **[ACL-2018]** Aspect Based Sentiment Analysis with Gated Convolutional Networks [[paper]](https://www.aclweb.org/anthology/P18-1234/)

<table>
    <tr>
        <td ><center><img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/Aspect%20Based%20Sentiment%20Analysis%20with%20Gated%20Convolutional%20Networks/GCAE_%E5%9B%BE/image-01.png" style="zoom:50%;" />
        </center></td>
        <td ><center><img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/Aspect%20Based%20Sentiment%20Analysis%20with%20Gated%20Convolutional%20Networks/GCAE_%E5%9B%BE/image-02.png" style="zoom:50%;" />
        </center></td>
    </tr>
</table>



- **[EMNLP-2018]** Multi-grained Attention Network for Aspect-Level Sentiment Classification [[paper]](https://www.aclweb.org/anthology/D18-1380/)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/MGAN%E5%9B%BE/image-01.png" style="zoom:50%;" />

------

### E2E-ABSA

- **[AAAI-2019]** A Uniﬁed Model for Opinion Target Extraction and Target Sentiment Prediction [[paper]](https://arxiv.org/abs/1811.05082)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/E2E-TBSA%E5%9B%BE/image-03.png" style="zoom:50%;" />



- **[EMNLP-2019]** Exploiting BERT for End-to-End Aspect-based Sentiment Analysis [[paper]](https://arxiv.org/abs/1910.00883)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/BERT-E2E-ABSA.assets/image-02.png" style="zoom:50%;" />

------

### Complete ABSA

- **[AAAI-2020]** Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis [[paper]](https://arxiv.org/abs/1911.01616v2)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/What-How-Why.assets/image-02.png" style="zoom:50%;" />

------

### ATE

- **[IJCAI-2018]** Aspect Term Extraction with History Attention and Selective Transformation [[paper]](https://arxiv.org/abs/1805.00760)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/HAST%E5%9B%BE/image-02.png" style="zoom:50%;" />

------

### ATE & OTE

- **[AAAI-2017]** Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms [[paper]](http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/ABSA%E8%AE%BA%E6%96%87%E6%80%BB%E7%BB%93.assets/image-01.png" style="zoom:80%;" />

------

### Emotion Classiﬁcation

- **[EMNLP-2018]** Improving Multi-label Emotion Classiﬁcation via Sentiment Classiﬁcation with Dual Attention Transfer Network [[paper]](https://www.aclweb.org/anthology/D18-1137/)

<img src="https://chenhao-1300052108.cos.ap-beijing.myqcloud.com/ch-Pic/NLP/NLP%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/DATN%E5%9B%BE/image-08.png" style="zoom:50%;" />

------

## Projects & Competition

- [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch)
- 2018 CCF-汽车行业用户观点主题及情感识别ASC挑战赛
  - Rank 1 [[code]](https://github.com/yilifzf/BDCI_Car_2018)
  - Rank 7 [[code]](https://github.com/nlpjoe/CCF-BDCI-Automotive-Field-ASC-2018)
- 2018 AI Challenger 全球AI挑战赛 - 细粒度用户评论情感分析
  - Rank 1 [[code]](https://github.com/chenghuige/wenzheng) [[link]](https://tech.meituan.com/2019/01/25/ai-challenger-2018.html)
  - Rank 16 [[code]](https://github.com/xueyouluo/fsauor2018)
  - Rank 17 [[code]](https://github.com/BigHeartC/Al_challenger_2018_sentiment_analysis)

------

## DataSet

**Chinese**

- AI-Challenge [[data](https://drive.google.com/file/d/1OInXRx_OmIJgK3ZdoFZnmqUi0rGfOaQo/view)] 
- SemEval ABSA 2016 [[data](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)] 

**English**

- Amazon product data [[data](http://jmcauley.ucsd.edu/data/amazon/)]
- Web data: Amazon reviews [[data](https://snap.stanford.edu/data/web-Amazon.html)]
- Amazon Fine Food Reviews [[kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews)]
- SemEval ABSA

------

## Reference

- [jiangqn/Aspect-Based-Sentiment-Analysis](https://github.com/jiangqn/Aspect-Based-Sentiment-Analysis)
- [YZHANG1270/Aspect-Based-Sentiment-Analysis](https://github.com/YZHANG1270/Aspect-Based-Sentiment-Analysis)
- [Data-Competition-TopSolution](https://github.com/Smilexuhc/Data-Competition-TopSolution)