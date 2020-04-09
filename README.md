# Recommender Systems Effect on Content Diversity
## An ABM approach to estimate information filter effects on cultural markets diversity.


### Working paper -


For access to the working paper please access [click here](https://www.researchgate.net/publication/340539715_Evaluating_Recommender_Systems_effect_on_Content_Diversity_An_ABM_framework_Working_Paper).


This work has been accepted for presentation at the ["Complexity and Policy Studies 2020"](https://www.caps-conference.org/)
conference this year. 


### Introduction - 
Digital markets depend on information filters that make the massive flow of interaction manageable for suppliers, distributors, and consumers. Beyond this operational cornerstone, information filters also have a passive role on how markets organize and behave. The design of these filters has a significant role on who gets what, funneling consumers into few suppliers or promoting obscure products and services that can better satisfy consumer needs. The potential effect of recommendations appears to be larger on cultural or entertainment and media industries, where product’s uncertainty is usually high. As such, cultural diversity and market concentration on content platforms (e.g. Google News, YouTube, Spotify) are susceptible to the effect of recommendation systems algorithms. 


The study of diversity has been a focal topic for individual recommendation optimizations, but little attention has been given to aggregated measures of diversity. Previous work on this area states that collaborative based recommender systems have an impact on sales diversity. I expand on this presenting an agent-based model to test the impacts of state-of-the-art recommender systems on cultural markets. Using heterogeneous agents to represent user behavior and specific market situations, the model works as a framework to estimate the effects of recommender system algorithms on sales and cultural diversity. Early results confirm previous work on the effect of collaborative filtering methods on diversity. The next step of this project includes the use of machine learning algorithms. Future work will provide useful insights for marketing modeling, content platform policy, and the use of information filters in market design. 


### Model overview, design, and details -

#### Overview.
The model represents a population of users that search for optimal experiences in an uncertain market of cultural products. 

#### Design.
There are 3 main agents in the model:

1. Market agent:
This object represents the market environment where Users interact with Products and Information Filters. The Information Filter mechanism is executed in the market agent. Thus, User agents receive recommendations from the Market when a filter is active. Following this design, the market represent a single platform where all recommendations are centralized (e.g. YouTube, Spotify, etc.).

2. User agent:
The user agent represents individuals that search and attempt to evaluate the utility of new experience products. Users search every time they are activated in the simulation steps. They search for a new product, estimate the potential utility, and then consume the best prospect. When a filter is activated, users search from the pool of products that have been recommended.

3. Product agent: 
A passive object that represents each item of the available library of products (e.g. movies in Netflix). Their key properties are product features, ratings, and number of times viewed.

Table 1.
| Agent         | Parameters           | Type  |  Range     |  Initialization values |
| ------------- |:--------------------:|:-----:|:----------:|:----------------------:|
| Market        | User population      | Int   |[2,40]*     |    [2,5,10,20,40]*     |
|               | Product space        | Int   |[0.4,8]*    |    [0.4,1,2,4,8]*      |
|               | Filter type          | Str   |['None','Cognitive','Sociological']|   ['None','Cognitive','Sociological'] |
*thousands


#### Details

The model was developed and simulated using Python 3.7.4 with Spyder 3.3.6 in 64-bit processors with Windows 10.
Simulations for early results and verification were executed for the following scenarios.

The code is available [here](https://github.com/Andrelhu/Diversity-on-RecSys/RecSys_ABM_Model.py).

### Results

Data on the repository is available to [explore early results](https://github.com/Andrelhu/Diversity-on-RecSys/SimResults) of the simulations.
