---
date: 2025-08-21
title: "Anomaly Detection: Generalized Extreme Studentized Deviate Test"
draft: false
toc: false 
---
<div style="margin-bottom: 48px;">

<a href="https://www.stat.cmu.edu/technometrics/80-89/VOL-25-02/v2502165.pdf" download style="background-color:#007BFF; color:#fff; padding:10px 15px; text-decoration:none; border-radius:5px; margin-right:20px; display:inline-block;">
  📥 Download Rosner 1983 PDF
</a>

</div>

## Detecting the Unusual: A look at GESD or Rosner's Test for Outlier Detection

In an era where data drives nearly every decision, the ability to spot what *doesn't* fit has become more critical than ever. Whether it's detecting fradulent transactions, monitoring network security, identifying equipment failures, or ensuring product quality, anomaly detection serves as a vital safguard. By uncovering patterns in data that do not conform to what is normal or expected, it enables us to respond quickly to risks, reduce losses, and even acticipate problems before they escalate.

But anomalies aren’t always obvious. Many statistical and machine learning techniques are sensitive to the presence of outliers or “contamination” in the data. For instance, simple summary statistics like the mean and standard deviation can be distorted by just a single extreme (and inaccurate) value. Averages shift, variances inflate, and models trained on such contaminated data often perform poorly. This is why checking for outliers is a routine—and crucial—part of any analysis.

The challenge, of course, is this: how do you decide what is “unusual” when the only thing you have is your dataset itself?

## What is Anomaly Detection?

Anomaly detection, also known as outlier detection or novelty detection, is the process of identifying data points that deviate significantly from the majority. These unusual points might represent:

* A fraudulent transaction hidden among millions of legitimate ones,

* A malfunctioning sensor in a manufacturing line,

* A sudden spike in network traffic indicating a security breach, or

* A simple recording error in a dataset.

Techniques for anomaly detection span a wide range:

* Visual methods: Boxplots, scatterplots, and histograms can quickly highlight outliers.

* Distance-based methods: Algorithms such as nearest neighbors or clustering methods flag points that are “far” from the rest.

* Statistical approaches: Techniques like Z-scores, Grubbs’ test, and GESD (Rosner’s test) quantify how extreme a value is relative to the rest of the distribution.

* Machine learning approaches: Isolation forests, autoencoders, and deep learning models can be used for complex, high-dimensional data.

Each method has trade-offs, and without rigorous subject matter expertise many involve some amount of arbitrary decision-making or heuristics. For example, deciding how many standard deviations from the mean qualifies as “too far,” or choosing a distance threshold in clustering, is often subjective. These choices can vary depending on context and may lead to inconsistent results.

## GESD (Rosner’s Generalized Extreme Studentized Deviate Test)


### Step 1: Defining our Hypothesis Test
Let’s say we have $n$ observations, $x_1, x_2, ... , x_n$. Given an upper bound on the maximum number of outliers, $k$, we perform $k$ separate tests. A test for one outlier, a test for two outliers, ..., up to a test for $k$ outliers. Importantly, *$k$* is just the **maximum possible number of outliers you’re willing to test for** — it is not a commitment that there are exactly *$k$* outliers.

### Step 2: Remove the most extreme point, recompute, and repeat
The test works iteratively:  
- Compute the mean and (sample) standard deviation of the dataset.
- Find the most extreme value (the one farthest from the mean).  
- Calculate it's Studentized Deviate statistics, $R_k$. This statistic is just a standardized distance.  
- Remove the extreme point from the data and repeat this process on the reduced dataset. Each step we are recalculating a new mean and (sample) standard deviation.

Formally:  
- Let $x^*_1, x^*_2, ... , x^*_{n-i}$ denote the $n-i$ observations after removing $i$ most extreme points.
- The mean after removing $i$ extreme points is:  
$$
\bar{x}_{(i)} = \frac{1}{n-i} \sum_{j=1}^{n-i} x_j^*
$$

- The (sample) standard deviation of those remaining points is:  
$$
s_{(i)} = \sqrt{\frac{1}{n-i-1} \sum_{j=1}^{n-i} \big(x_j^* - \bar{x}_{(i)}\big)^2}
$$ 

- The “extreme value” at step $i$ is whichever observation is farthest from the current mean:  
$$
x_{(i)} = \max_{j=1, …, n-i} \big|x_j^* - \bar{x}_{(i)}\big|
$$ 

- The test statistic is:  

$$
R_{i} = \frac{|x_{(i)} - \bar{x}_{(i)}|}{s_{(i)}}
$$  

### Step 3: Compare against critical values
Each $R_i$ is compared to a critical threshold $\lambda_i$ derived from the Student’s $t$-distribution:

$$
\lambda_{i} = \frac{(n-i) \, t_{p, n-i-1}}{\sqrt{(n-i-1 + t_{p, n-i-1}^2)(n-i+1)}}
$$  

where  

$$
p = 1 - \frac{\alpha}{2(n-i+1)}
$$  

and $\alpha$ is your significance level (commonly 0.05).  

### Step 4: Decide how many outliers exist
- The number of outliers is determined by finding the largest $i$ such that $R_i > \lambda_i$.


## Demonstrating an implementation

Here is a function implementing Rosner's test in Python:


```python
import numpy as np
from scipy.stats import t
def rosner_outliers(data, max_outliers = 1, alpha = 0.1, verbose = False):
    """
    Generalized Extreme Studentized Deviate Test (Rosner 1983) for Outliers.

    Parameters
    ----------
    data : 1D data
        Data to test. NaNs are ignored in calculations and returned as False (non-outlier).
    max_outliers : int
        Maximum number of outliers to test for (k).
    alpha : float
        Significance level.

    Returns
    -------
    extreme_idx : list of int
        Indices of the candidate extreme values in the original data (ordered by 
        extremeness).
    Rs : list of float
        The computed test statistics R for each candidate extreme value.
    lambdas : list of float
        The corresponding critical values (λ) used to determine significance.
    outliers : ndarray of bool, shape (n,)
        Boolean mask indicating which points in `data` are considered outliers. 
        True for detected outliers, False otherwise (including NaNs).

    References
    ----------
    Rosner, B. (1983). "Percentage Points for a Generalized ESD Many-Outlier 
    Procedure." Technometrics, 25(2), 165–172.
    """
    # Making 1D data into an np array
    data = np.array(data)
    n = len(data)

    # Given j corresponding to the jth most extreme value then:
    # extreme_idx[j-1] is the index of the jth data value on the original input data
    # Rs[j-1] is the corresponding R value of the jth extreme data point
    # lambdas[j-1] is the corresponding lambda value of the jth extreme data point
    extreme_idx = [] # Array of size k containing indices of original data
    Rs          = [] # Array of size k with Rs
    lambdas     = [] # Array of size k with lambdas

    exclusion_idx = np.isnan(data) # init mask - automatically excludes NaN
    for i in range(1, max_outliers + 1):
        temp_idx = np.where(~exclusion_idx)[0]   # indices of current "kept" data
        temp_data = data[temp_idx]
       
        mu = np.mean(temp_data)
        sd = np.std(temp_data, ddof = 1) # Sample standard deviation

        # Find R_i
        deviations = np.abs(temp_data - mu)
        max_dev_idx = np.argmax(deviations)
        Rs.append(deviations[max_dev_idx] / sd)

        # Tracking observation index
        extreme_idx.append(temp_idx[max_dev_idx])

        # Find lambda_i
        p = 1 - alpha / ( 2 * (n - i + 1) )  
        t_crit = t.ppf(p, df=n-i-1)
        lambdas.append( (n-i)*t_crit / np.sqrt( (n-i+1)*(n-i-1+t_crit**2) ) )

        # Marking point as checked
        exclusion_idx[temp_idx[max_dev_idx]] = True

    # Finding largest index where Rs[i] > lambdas[i]
    i_keep = -1
    for i in range(max_outliers):
        if Rs[i] > lambdas[i]:
            i_keep = i
    # Create outlier mask
    outliers = np.zeros(n, dtype=bool)
    if i_keep >= 0:
        for j in range(i_keep+1):
            outliers[extreme_idx[j]] = True

    # Printing verbose report
    if verbose:
        print()
        print("H0:  there are no outliers in the data")
        print(f"Ha:  there are up to {max_outliers} outliers in the data")
        print()
        print(f"Significance level:  \u03B1 = {alpha:.2g}")
        print("Critical region:  Reject H0 if Ri > critical value")
        print()
        print("Summary Table for Two-Tailed Test")
        print("---------------------------------------")
        print("      Exact           Test     Critical  ")
        print("  Number of      Statistic    Value, \u03BBi")
        print(f"Outliers, i      Value, Ri     {100 * alpha:>5.0f} %  ")
        print("---------------------------------------")
        for i in range(max_outliers):
            star = " *" if Rs[i] > lambdas[i] else "  "
            print(f"{i+1:10d}{Rs[i]:14.3f}{lambdas[i]:12.3f}{star}")
        print()

    return extreme_idx, Rs, lambdas, outliers
```

The Rosner (1983) paper uses the following data as an example:


```python
x = np.array([float(x) for x in "-0.25 0.68 0.94 1.15 1.20 1.26 1.26 1.34 1.38 1.43 1.49 1.49 \
          1.55 1.56 1.58 1.65 1.69 1.70 1.76 1.77 1.81 1.91 1.94 1.96 \
          1.99 2.06 2.09 2.10 2.14 2.15 2.23 2.24 2.26 2.35 2.37 2.40 \
          2.47 2.54 2.62 2.64 2.90 2.92 2.92 2.93 3.21 3.26 3.30 3.59 \
          3.68 4.30 4.64 5.34 5.42 6.01".split()])
```


```python
idx, Rs, lambdas, outliers = rosner_outliers(x, max_outliers = 10, alpha = 0.05, verbose=True)
```

    
    H0:  there are no outliers in the data
    Ha:  there are up to 10 outliers in the data
    
    Significance level:  α = 0.05
    Critical region:  Reject H0 if Ri > critical value
    
    Summary Table for Two-Tailed Test
    ---------------------------------------
          Exact           Test     Critical  
      Number of      Statistic    Value, λi
    Outliers, i      Value, Ri         5 %  
    ---------------------------------------
             1         3.119       3.159  
             2         2.943       3.151  
             3         3.179       3.144 *
             4         2.810       3.136  
             5         2.816       3.128  
             6         2.848       3.120  
             7         2.279       3.112  
             8         2.310       3.103  
             9         2.102       3.094  
            10         2.067       3.085  
    


Here are the points detected as outliers:


```python
x[outliers]
```




    array([5.34, 5.42, 6.01])



Note that the data does not need to be ordered and that NaNs are ignored if present:


```python
np.random.seed(2030)
z = np.array([float(x) for x in "-0.25 0.68 0.94 1.15 1.20 1.26 1.26 1.34 1.38 1.43 1.49 1.49 \
          1.55 1.56 1.58 1.65 1.69 1.70 1.76 1.77 1.81 1.91 1.94 1.96 \
          1.99 2.06 2.09 2.10 2.14 2.15 2.23 2.24 2.26 2.35 2.37 2.40 \
          2.47 2.54 2.62 2.64 2.90 2.92 2.92 2.93 3.21 3.26 3.30 3.59 \
          3.68 4.30 4.64 5.34 5.42 6.01 nan nan nan".split()])
shuffler = np.random.permutation(len(x))
z = z[shuffler]
print(z)
idx, Rs, lambdas, outliers = rosner_outliers(z, max_outliers = 10, alpha = 0.05, verbose=True)
print("Outliers:", z[outliers])
```

    [ 1.15  2.47  1.34  3.3   2.06  2.62  1.7   6.01  1.2   1.49  3.68  1.56
      1.69  2.23  2.93  4.64  1.38  2.64  1.26  4.3   5.34  2.26  0.68  1.96
      1.81  1.65  1.58  2.92  3.21  1.91  1.94  2.4   1.76  3.26  1.49  2.24
     -0.25  0.94  1.43  2.15  2.9   2.09  2.54  1.99  2.37  2.35  2.92  5.42
      2.14  3.59  2.1   1.55  1.77  1.26]
    
    H0:  there are no outliers in the data
    Ha:  there are up to 10 outliers in the data
    
    Significance level:  α = 0.05
    Critical region:  Reject H0 if Ri > critical value
    
    Summary Table for Two-Tailed Test
    ---------------------------------------
          Exact           Test     Critical  
      Number of      Statistic    Value, λi
    Outliers, i      Value, Ri         5 %  
    ---------------------------------------
             1         3.119       3.159  
             2         2.943       3.151  
             3         3.179       3.144 *
             4         2.810       3.136  
             5         2.816       3.128  
             6         2.848       3.120  
             7         2.279       3.112  
             8         2.310       3.103  
             9         2.102       3.094  
            10         2.067       3.085  
    
    Outliers: [6.01 5.34 5.42]

