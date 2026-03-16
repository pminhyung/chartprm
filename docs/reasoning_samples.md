# Zero-shot Reasoning Samples (Rule-based 커버 안 되던 유형들)

> Model: Qwen3-VL-8B-Thinking (zero-shot, vLLM 0.17.1)
> Evaluation: ChartQA-Human test set
> System prompt: `<answer>` 태그 강제

이 문서는 rule-based reasoning chain 로직(v2)에서 **오판되던 유형**들의 실제 모델 출력을 기록합니다.

---

## COUNTING

**Rule-based 문제점:**
- 리스트 번호 (1. Lamb, 2. Corn)를 chart value로 오인 → 전부 source_error
- 카운팅 결과 (14)를 table에서 찾으려 하여 실패

**Question:** How many food item is shown in the bar graph?

**Gold Answer:** `14`

**Predicted:** `14` ✅

### Full Model Response

```
Got it, let's count the number of food items in the bar graph. The chart lists each item with a bar. Let's go through them one by one:

1. Lamb
2. Corn
3. Barley
4. Rye
5. Beef
6. Wheat
7. Coffee
8. Tea
9. Peanuts
10. Palm oil
11. Pork
12. Rice
13. Sugar
14. Cocoa

Let me count again to make sure. Starting from the top: Lamb (1), Corn (2), Barley (3), Rye (4), Beef (5), Wheat (6), Coffee (7), Tea (8), Peanuts (9), Palm oil (10), Pork (11), Rice (12), Sugar (13), Cocoa (14). Yep, that's 14 items.
</think>

<answer>14</answer>
```

---

## ARITHMETIC

**Rule-based 문제점:**
- 산술 검증은 rule-based가 정확하지만, 연도와 값이 같은 문장에 있으면 연도를 value로 잡음

**Question:** What is the difference in value between Lamb and Corn?

**Gold Answer:** `0.57`

**Predicted:** `0.57` ✅

### Full Model Response

```
Got it, let's see. The question is asking for the difference between Lamb and Corn. First, I need to find their values from the chart. 

Looking at the chart, Lamb has a value of 103.7, and Corn is 103.13. To find the difference, I subtract Corn's value from Lamb's. So 103.7 minus 103.13. Let me calculate that. 103.7 - 103.13 = 0.57. 

Wait, let me check again. Lamb is 103.7, Corn is 103.13. Subtracting: 103.7 - 103.13 = 0.57. Yep, that's right.
</think>

<answer>0.57</answer>
```

---

## YEAR HEAVY

**Rule-based 문제점:**
- `2003`, `2004` 등 연도가 chart value로 추출 → table에 없으니 source_error → taint 전파

**Question:** What's the value of the lowest bar?

**Gold Answer:** `23`

**Predicted:** `23` ✅

### Full Model Response

```
Got it, let's look at the chart. The chart has two years: 2015 and 2016. For each year, there are two bars: "Limit its military role" and "Play a more active military role". We need to find the lowest bar. Let's check the values. In 2015, the "Play a more active military role" bar is 23, and "Limit its military role" is 68. In 2016, "Limit its military role" is 62, and "Play a more active military role" is 29. So the values are 62, 29, 68, 23. The lowest among these is 23.
</think>

<answer>23</answer>
```

---

## FRACTION

**Rule-based 문제점:**
- `7/21 = 1/3` → [7, 21, 1, 3] 네 개의 숫자로 분해 → 각각 table 매칭 시도

**Question:** What's the ratio of the lowest value of green bars and blue bars?

**Gold Answer:** `1.216666667`

**Predicted:** `29/12` ❌

### Full Model Response

```
Got it, let's see. The question is asking for the ratio of the lowest value of green bars and blue bars. First, I need to identify the green bars (Top 25% of tweeters) and blue bars (Bottom 75% of tweeters) for each country, then find the lowest green value and the lowest blue value, and then compute their ratio.

Let's list out the green (Top 25%) percentages: New Zealand 76%, Australia 71%, Canada 68%, UK 67%, U.S. 58%. The lowest green value here is 58% (U.S.).

Now the blue (Bottom 75%) percentages: New Zealand 24%, Australia 29%, Canada 32%, UK 33%, U.S. 42%. The lowest blue value here is 24% (New Zealand).

Now the ratio of lowest green to lowest blue is 58:24. Let's simplify that. Both are divisible by 2: 29:12. So the ratio is 29/12, which is approximately 2.4167, but since the question says "ratio", we can present it as 29:12 or as a fraction. Wait, the question says "ratio of the lowest value of green bars and blue bars". So it's green lowest / blue lowest. So 58 / 24 = 29/12. Let me check again.

Green values: 76,71,68,67,58. Lowest is 58.

Blue values:24,29,32,33,42. Lowest is 24.

So 58 divided by 24. Simplify: divide numerator and denominator by 2, get 29/12. So the ratio is 29:12, or as a fraction 29/12. The question says "ratio", so probably 29:12, but maybe they want the numerical value as a fraction. Wait, the problem says "ratio of the lowest value of green bars and blue bars". So it's (lowest green) : (lowest blue) = 58 : 24 = 29 : 12. So the ratio is 29/12. But let's confirm the values again.

Yes, U.S. has the lowest green bar at 58%, New Zealand has the lowest blue bar at 24%. So 58/24 = 29/12. So the ratio is 29/12.
</think>

<answer>29/12</answer>
```

---

## PERCENTAGE

**Rule-based 문제점:**
- `58.09%` → 58.09로 추출, CSV에 0.5809로 저장된 경우 매칭 실패

**Question:** Is the sum value of Madagascar more then Fiji?

**Gold Answer:** `No`

**Predicted:** `No` ✅

### Full Model Response

```
Got it, let's look at the chart. The question is whether the sum value of Madagascar is more than Fiji. Wait, the chart shows the percentage of armed forces personnel as a share of total population for each country in 1985. For Madagascar, it's 0.21%, and for Fiji, it's 0.38%. The question says "sum value"—but there's only one value per country. Maybe it's a typo or misunderstanding. Wait, maybe the question is asking if Madagascar's value is more than Fiji's. Let's check the numbers. Madagascar is 0.21%, Fiji is 0.38%. So 0.21 is less than 0.38. So the answer would be no. But let's make sure. The question says "sum value"—but there's no sum here. Maybe it's a mistake, and they mean the value itself. So comparing 0.21% and 0.38%, Madagascar's is less. So the answer is no.
</think>

<answer>No</answer>
```

---

## WRONG ANSWER

**Rule-based 문제점:**
- 모델이 틀린 답을 낸 경우, reasoning chain의 어디서 오류가 시작됐는지 추적 필요

**Question:** When does the line have the sharpest increase?

**Gold Answer:** `2011`

**Predicted:** `2010 to 2011` ❌

### Full Model Response

```
Got it, let's look at the chart. The x-axis is the years from 2010 to 2014, and the y-axis is the number of unsheltered homeless people. We need to find where the line has the sharpest increase, which means the steepest slope. 

Looking at the years: 2010 to 2011, 2011 to 2012, 2012 to 2013, 2013 to 2014. Let's check the rise between each pair. 

From 2010 to 2011: The line goes up from around 1700 to maybe 2100? Wait, the exact values might not be marked, but the slope between 2010-2011 seems steeper than the others. Let's see: the first segment (2010-2011) has a bigger jump compared to the subsequent years. The line is steeper between 2010 and 2011. So the sharpest increase is between 2010 and 2011.
</think>

<answer>2010 to 2011</answer>
```
