# Project Assessment: Week 3
To assess your project work, you should be able to answer the following questions:

## For query classification:

**How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 1000? To 10000?**

Using 1000 as the minimum reduced the number of categories to 151, this is a lot fewer than the 400 the project instructions suggested, so I believe I may have had a more aggressive pruning strategy. Rather than doing what the instructions seem to suggest, it rolls up any category with fewer than the minimum queries in no particular order.

```
The order in which you roll up categories below the threshold is up to you, but a good idea is to iteratively pick the category with the fewest queries, breaking ties arbitrarily. But again, it’s up to you.
```

When setting the minimum number of queries to 10000, the remaining category count is 11.

**What were the best values you achieved for R@1, R@3, and R@5? You should have tried at least a few different models, varying the minimum number of queries per category, as well as trying different fastText parameters or query normalization. Report at least 2 of your runs.**

Parameters
| model_no | min_queries | normalize | final_cat_count | epochs | learning_rate |
| --- | --- | --- | --- | --- | --- | 
| 1 | 1 | False |  | 5 | 0.2 |
| 2 | 1 | True | 1479 | 5 | 0.2 |
| 3 | 250 | True | 504 | 5 | 0.2 |
| 4 | 500 | True | 369 | 5 | 0.2 |
| 5 | 500 | True | 369 | 10 | 0.2 |
| 6 | 1500 | True | 270 | 10 | 0.2 |
| 10 | 10000 | True | 250 | 5 | 0.2 |
| 11 | 15000 | True | 238 | 5 | 0.2 |
| 12 | 15000 | False | | 5 | 0.2 |

Results

| model_no | p@1 | r@1 | p@3 | r@3 | p@5 | r@5 |
| --- | --- | --- | --- | --- | --- | --- | 
| 1 | X | X | X | X | X | X | 
| 2 | 0.562 | 0.562 | 0.252 | 0.757 | 0.164 | 0.82 | 
| 3 | 0.569 | 0.569 | 0.254 | 0.763 | 0.166 | 0.828 |
| 4 | 0.582 | 0.582 | 0.258 | 0.775 | 0.167 | 0.837 | 
| 5 | 0.583 | 0.583 | 0.259 | 0.776 | 0.168 | 0.838 |
| 6 | 0.614 | 0.614 | 0.269 | 0.807 | 0.172 | 0.862 |
| 10 | 0.654 | 0.654 | 0.275 | 0.826 | 0.175 | 0.876 |
| 11 | 0.662 | 0.662 | 0.277 | 0.830 | 0.176 | 0.880 |
| 12 | 0.665 | 0.662 | 0.280 | 0.841 | 0.178 | 0.890 |

1 would take an estimated 1 hour to run, so I didn't run it. I assume it would perform worse than 2.

## For integrating query classification with search:

For these results, I used model version 4 and I used `k=3` when predicting, to get the top 3 labels predicted by the model, and then only used the label for filtering or boosting if the predicted probability for that label was greater than 0.3. 

I decided to keep boosting low because there are so many categories so I didn't want an incorrect prediction to have an outsized effect on the ranking.

**Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.**

onkyo
predicted labels ((abcat0202003, 0.45), (pcmcat167300050040, 0.34))
baseline returned 54 results and max score was 0.060852595
filtering returned 29 results and max score was 0.060852595
boosting returned 477 results and max score was 0.060860094

sims 3 mac
predicted labels (pcmcat174700050005, 0.7) (seems to be the code for laptops)
baseline returned 3613 results and max score was 839.2068
filtering returned 490 results and max score was 301.77863
boosting returned 5376 results and max score: 301.80746
Filtering seemed to narrow it down incorrectly since it focused on the laptop part rather than the game part. Even though boosting did expand the search a lot the top result was still "correct"

xbox charger
predicted labels (abcat0715001, 0.82)
baseline returned 3251 results and max score was 0.15024781
filtering returned 313 results and max score was 0.0832135
boosting returned 3352 results and max score was 0.18160768
the baseline focused more on the charger, but with the filter it focused more on the xbox. Boosting helped.  

nikon d90
predicted labels (pcmcat180400050000, 0.66)
baseline returned 454 results and max score was 0.13394433
filtering returned 11 results and max score was 0.13394433

**Give 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.**

Filtering seemed to hurt in the cases where it was a very high level category that was predicted

its always sunny in philadelphia
(cat02015, prob 0.98) (this seems to be a very high level category)
baseline returned 169 results and max score was 1.3147714e-05
filtering returned nothing
boosting returned similar results

shrek
(cat02015, 0.32)(this seems to be a very high level category)
baseline returned 105 and max score was 0.060541872 (top results are games)
filtering returned nothing
boosting returned 507 results and max score was 0.1620115 (top result was the same)