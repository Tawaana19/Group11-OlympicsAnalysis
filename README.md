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

