# 1. get data (in df) from enemies.json, actions.json and effects.json
# 2. calc strength of each action based on damage, hit chance and effects
# 2.1 convert effects.json so the mapping of rating and name can be used on the effects an action can contain
# 2.2 calculate the average, max, min and spread of damage dices for the action of the enemy
# 2.3 combine the effects, and the values from the step before to get the strengt lvl of the action
# 3. join strength on each action of the enemies
# 4. find out the "optimal" action of an enemy
# 5. combine stats of enemy and the best action/s for final danger level of the enemy