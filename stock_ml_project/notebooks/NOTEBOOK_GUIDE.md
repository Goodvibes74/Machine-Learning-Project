# 📓 Jupyter Notebooks Guide

## What are Jupyter Notebooks?

Jupyter notebooks are **interactive documents** that combine:
- Code (that you can run)
- Text explanations (in Markdown)
- Visualizations (plots and charts)
- Output results

Think of them as an **interactive textbook** where you can:
- Run code cell-by-cell
- See results immediately
- Modify and experiment
- Learn by doing

---

## 🚀 Getting Started with Notebooks

### Step 1: Install Jupyter

Already done if you installed from requirements.txt!

```bash
pip install jupyter
```

### Step 2: Start Jupyter

```bash
# Navigate to your project folder
cd stock-prediction

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Start Jupyter
jupyter notebook
```

This will:
1. Start a local server
2. Open your browser automatically
3. Show a file navigator

### Step 3: Open a Notebook

In the browser:
1. Navigate to `notebooks/` folder
2. Click on `01_data_exploration.ipynb`
3. The notebook opens in a new tab!

---

## 📚 Notebook Structure

We have **4 notebooks** that build on each other:

### 1️⃣ Data Exploration (`01_data_exploration.ipynb`)
**Time:** 30-45 minutes

**What you'll do:**
- Download stock data
- Understand OHLCV format
- Create basic visualizations
- Calculate returns
- Analyze up/down days

**Start here if:** You're new to stock data

### 2️⃣ Feature Engineering (`02_feature_engineering.ipynb`)
**Time:** 45-60 minutes

**What you'll do:**
- Create moving averages
- Calculate volatility
- Build technical indicators
- Create sentiment proxy
- Prepare data for ML

**Start here if:** You understand stock data basics

### 3️⃣ Model Training (`03_model_training.ipynb`)
**Time:** 45-60 minutes

**What you'll do:**
- Train Random Forest model
- Evaluate performance
- Understand confusion matrices
- Check feature importance
- Compare with baseline

**Start here if:** You have feature data ready

### 4️⃣ Backtesting (`04_backtesting.ipynb`)
**Time:** 60 minutes

**What you'll do:**
- Walk-forward validation
- Test model stability
- Compare strategies
- Statistical significance
- Simulate trading

**Start here if:** You want to validate properly

---

## 🎮 How to Use Notebooks

### Running Cells

**Keyboard shortcuts:**
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `Alt + Enter`: Run cell and insert new below

**Or use the toolbar:**
- Click ▶️ button to run current cell
- Use "Run All" to execute entire notebook

### Cell Types

**Code cells** (gray background):
```python
# This is code - you can edit and run it!
print("Hello World")
```

**Markdown cells** (white background):
- Explanatory text
- Can't be "run" (just rendered)
- Contains instructions

### Editing Code

1. **Click on a code cell** to edit it
2. **Modify the code** as you like
3. **Run it** with Shift+Enter
4. **See the output** below the cell

### Example Workflow

```python
# Original code
ticker = 'AAPL'

# You can change it to:
ticker = 'MSFT'  # Try Microsoft instead!

# Then run the cell to see Microsoft's data
```

---

## 💡 Learning Strategy

### For Beginners (New to ML)

**Week 1:**
- Day 1: Notebook 1 (Data Exploration)
- Day 2: Notebook 1 exercises
- Day 3: Notebook 2 (Feature Engineering)
- Day 4: Notebook 2 exercises
- Day 5: Review and practice

**Week 2:**
- Day 1: Notebook 3 (Model Training)
- Day 2: Notebook 3 experiments
- Day 3: Notebook 4 (Backtesting)
- Day 4: Complete exercises
- Day 5: Try different stocks

### For Intermediate (Some ML Experience)

**Day 1:** Notebook 1 + 2
**Day 2:** Notebook 3
**Day 3:** Notebook 4
**Day 4-5:** Experiments and modifications

### For Advanced (Experienced)

Run all 4 notebooks in sequence (3-4 hours), then:
- Modify hyperparameters
- Add new features
- Try different models
- Implement improvements

---

## 🔧 Tips & Tricks

### Save Your Work
- Notebooks **auto-save** every few minutes
- Manually save: `Ctrl+S` or File → Save

### Restart Kernel
If something breaks:
1. Kernel → Restart & Clear Output
2. Run cells from top again

### Variables Persist
```python
# Cell 1
x = 10

# Cell 2 (later)
print(x)  # This works! x is still in memory
```

### Order Matters!
- Run cells **in order** (top to bottom)
- If you skip cells, variables might not exist

### Check What's Loaded
```python
# See all variables
%whos

# See just DataFrames
%whos DataFrame
```

---

## 🐛 Common Issues

### "NameError: name 'X' is not defined"
**Problem:** You skipped a cell that defined X

**Solution:** Run cells in order from the top

### "Module not found"
**Problem:** Missing package or wrong directory

**Solution:**

```python
# Add this at top of notebook
import sys

sys.path.append('')  # Go up one directory
```

### Plots not showing
**Problem:** Matplotlib backend issue

**Solution:** Add at top:
```python
%matplotlib inline
```

### Kernel keeps dying
**Problem:** Not enough memory or infinite loop

**Solution:**
- Restart kernel
- Use less data (smaller date range)
- Close other programs

---

## 🎯 Exercises in Notebooks

Each notebook has exercises marked:

```python
## 🎯 Exercise: Try It Yourself!

# YOUR CODE HERE
# Instructions...
```

**Don't skip these!** They're crucial for learning.

**Stuck on an exercise?**
1. Read the instructions carefully
2. Look at similar code above
3. Try something (even if wrong!)
4. Google the error message
5. Ask for help

---

## 📊 Understanding Output

### Text Output
```
✅ Success messages
⚠️ Warnings
❌ Errors
📊 Results
💡 Insights
```

### Plots
- Automatically display below code
- Interactive in some cases
- Can be saved (right-click → Save)

### DataFrames
- Show as formatted tables
- Only first/last rows by default
- Click to see more

---

## 🚀 Advanced Features

### Magic Commands

```python
# Time how long code takes
%timeit my_function()

# Run a system command
!ls  # Mac/Linux
!dir  # Windows

# Load external script
%run my_script.py
```

### Multiple Outputs

```python
# Show multiple things
print("Text output")
plt.plot([1,2,3])  # Plot
data.head()  # DataFrame

# All will display!
```

### LaTeX Math

```markdown
The formula is: $y = mx + b$

Or block format:
$$
accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$
```

---

## 📝 Best Practices

### 1. Add Your Own Notes
```python
# My observation: AAPL is more volatile than MSFT
# Question: Why does SMA_20 lag so much?
```

### 2. Experiment!
```python
# Original
window = 5

# Try different values
window = 10  # What happens?
window = 3   # How about this?
```

### 3. Document Changes
```markdown
## My Modifications
- Changed ticker from AAPL to TSLA
- Increased n_estimators to 200
- Result: Accuracy improved by 2%
```

### 4. Create Checkpoints
Before major changes:
- File → Save and Checkpoint
- Or duplicate the notebook

---

## 🎓 Learning Path

### Beginner Path
1. **Read** the markdown explanations
2. **Run** each code cell
3. **Observe** the outputs
4. **Experiment** with small changes
5. **Complete** the exercises

### Intermediate Path
1. **Skim** the explanations
2. **Run** all cells quickly
3. **Focus** on exercises
4. **Modify** parameters
5. **Compare** results

### Advanced Path
1. **Use** as reference
2. **Extract** useful code
3. **Build** your own analysis
4. **Extend** functionality
5. **Share** improvements

---

## 📚 Additional Resources

### Jupyter Shortcuts
Press `H` in notebook to see all shortcuts

**Essential shortcuts:**
- `A`: Insert cell above
- `B`: Insert cell below
- `D, D`: Delete cell
- `M`: Change to Markdown
- `Y`: Change to Code
- `Z`: Undo delete

### Helpful Links
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Markdown Guide](https://www.markdownguide.org/cheat-sheet/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

---

## ✅ Checklist

Before starting:
- [ ] Jupyter installed
- [ ] Virtual environment activated
- [ ] In correct directory
- [ ] Can open notebooks

While working:
- [ ] Run cells in order
- [ ] Read explanations
- [ ] Try exercises
- [ ] Experiment with code
- [ ] Save frequently

After completing:
- [ ] Understand the concepts
- [ ] Can explain to someone else
- [ ] Completed exercises
- [ ] Tried different parameters
- [ ] Made it your own!

---

**Happy Learning! 🎉**

Remember: The best way to learn is by **doing**. Don't just read the code—run it, break it, fix it, and make it better!
