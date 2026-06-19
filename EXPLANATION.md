Q1. What percentage of customers in your dataset have y = yes? What does this imbalance mean for how you'd evaluate a model?

Out of 45211 only 11.70% subscribed (y = yes), while 88.30% did not. This imbalance means accuracy Is a very misleading metric, if we use a random model that predicts no for every single costumer, it will still manage to get an accuracy of 88.30% despite being completely useless. So accuracy alone tells you almost nothing here. We have to look at precision, recall, and F1 on the minority class specifically, because that's the class that actually matters.


Q2. Which job category had the highest subscription rate? Does this make sense to you intuitively?

Students had the highest rate at 28.68%, followed by retired at 22.79%. Honestly, this surprised me a bit, but when I thought about it, it made sense for students, they're maybe taking out their first financial product or they may not have many other investment options. Retired people make even more sense to me, they have their savings and want something safe and fixed return rather than anything risky. Both groups are probably more attracted towards "put your money here and earn interest" pitch than someone mid-career juggling a mortgage and kids.


Q3. Which feature had the highest importance in your tree-based model? Why do you think that is?

poutcome_success came out on top at 0.1542. This means whether the customer subscribed in a previous campaign was the single strongest predictor. Once I saw it I couldn't unsee how obvious it is, if someone said yes before, they've already shown they're the type of person who buys this product. No demographic feature is going to beat that signal. contact_unknown came second at 0.1316 which was more surprising, and a bunch of specific months showed up in the top 10 too, which tells me timing matters more than I expected going in.


Q4. Why is F1 a better metric than accuracy for this particular dataset?

I actually saw this play out directly while building this. Before I added class_weight balanced, my XGBoost had 90.5% accuracy but was catching only 39% of actual subscribers — it was basically just betting on "no" most of the time and getting away with it because 88% of customers are no anyway. After fixing the imbalance, accuracy dropped to 87.8% but recall jumped to 81% and F1 went from 0.49 to 0.61. The model got worse on paper but massively better at its actual job. F1 is a better metric here because it forces you to care about both catching real subscribers and not spamming people with wrong predictions — accuracy lets you ignore all of that.


Q5. Pick one of your 5 sample predictions. Do you actually agree with the model's call?

Customer 2 is the interesting one. Age 35, balance 1215, call lasted 597 seconds, has a housing loan, no previous campaign success. Model predicted subscribe at 54.44% probability but it was wrong, he didn't actually subscribe. But I don't think the model made a bad call here, I think 54% was the honest answer. The long call duration pushed toward yes, the housing loan and no prior history pushed toward no, and those signals genuinely cancel each other out. A human RM looking at this customer beforehand probably wouldn't have been confident either. The model being uncertain on an actually uncertain customer feels right to me, what would worry me is if it was 95% confident and wrong.


Q6. What would likely break first if 200 RMs were hitting your /predict endpoint simultaneously? What's one thing you'd change?

The first thing to break would be the FastAPI/Uvicorn server because it is running with a single worker process. If around 200 RMs were sending requests to the `/predict` endpoint simultaneously, requests would begin to queue up, increasing response times and potentially causing timeouts for some users.The quickest improvement would be running multiple Uvicorn workers (e.g., `--workers 4`) so the server can handle multiple requests at the same time instead of processing them through a single worker.For larger-scale deployments, techniques such as rate limiting, caching repeated requests, and asynchronous processing could further improve performance, especially for endpoints that depend on external APIs.


Q7. What does the LLM explanation actually add over just showing a probability score?

A number like 0.9908 doesn't tell an RM anything about what to say when they pick up the phone. The explanation for that same customer: 64 year old retired, low balance, came back suggesting the RM highlight higher interest rates and ask about financial goals and risk tolerance. That's an actual conversation starter, not a data point. Most RMs aren't going to know what to do with a probability score, but they can immediately act on "this person is retired and likely looking for safe returns, lead with interest rates." The probability tells you whether to call. The explanation tells you how.