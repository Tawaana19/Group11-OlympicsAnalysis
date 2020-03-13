# ECE 143 Team 11 - Olympic Data Analysis

## Organization

dat folder: Contains all relevant .csv data files pre and post cleaning

src folder: Contains all .py files 

## Steps to Run

### GDP Calculations


All functions are located within src/Data_Exploration/GDP_functions

#### To call Scatter - Developing vs Developed
Firstly import using: 
<pre><code> from GDP_functions import medal_vs_gdp_scatter </code></pre>

To call: 

* argument 1,2 = Country code. 

* argument 3 = Type of medal count ('total', 'gold', 'silver', 'bronze')
 
* argument 4 = boolean to save or not

for example:  
<pre><code> plt = medal_vs_gdp_scatter('USA','USA','total',False)</code></pre>


#### To call Scatter - rate of changes
Firstly import using: 
<pre><code> from GDP_functions import medal_vs_gdp_scatter </code></pre>

Then run the function using:  
<pre><code> plt = plot_rate()</code></pre>

### Tokyo 2020 Predictions
All functions are located within src/Predictor

#### To call Stacked Bar Plot - Ratio of medals
Firstly import using: 
<pre><code> from prediction_plots import stacked_bar_plot </code></pre>

Then run the function using:  
<pre><code> stacked_bar_plot() </code></pre>

#### To call Medals line plots - Medals trends plots
Firstly import using: 
<pre><code> from prediction_plots import medal_trends_1 
<br> from prediction_plots import medal_trends_2</code></pre>

Then run the function using:  
<pre><code> medal_trends_1() 
<br> medal_trends_2() </code></pre>

#### To call World map - World map with medals
Firstly import using: 
<pre><code> from prediction_plots import world_map </code></pre>

Then run the function using:  
<pre><code> world_map() </code></pre>
This will open a .html page which consists of all the three world maps indicating the number of medals won.



