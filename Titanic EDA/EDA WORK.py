#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_df = pd.read_csv("C:/Users/jhala/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/jhala/Downloads/test.csv")


# In[3]:


print(f'The Training Dataset contains, Rows: {train_df.shape[0]} & Columns: {train_df.shape[1]}')
print(f'The Test Dataset contains, Rows: {test_df.shape[0]} & Columns: {test_df.shape[1]}')


# In[4]:


train_df.info()


# In[5]:


train_df.head().style.background_gradient(cmap='crest')


# In[6]:


numeric_features = train_df.select_dtypes(exclude=['object']).columns
numeric_features


# In[7]:


numeric_df = train_df[numeric_features]
numeric_df


# In[8]:


numeric_df.describe().T.style.background_gradient(cmap='Blues')


# In[9]:


corr_matrix = numeric_df.corr()
corr_matrix


# In[10]:


plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues');


# In[11]:


numeric_features_correlation_df = pd.DataFrame(numeric_df.corr().Survived)
abs(numeric_features_correlation_df).sort_values(by='Survived', ascending=False).style.background_gradient(cmap='Blues')


# In[12]:


size = list(numeric_df['Survived'].value_counts())
labels = ['Not Survived', 'Survived']
colors = ['Red', '#0da6ec']
explode = [0, 0.1]

def func(pct, allvals):
    absolute = int(round(pct/100*np.sum(allvals)))
    return "{:.1f}%\n({:d} Passengers)".format(pct, absolute)

plt.subplots(figsize=(8,8))
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, 
        autopct = lambda pct: func(pct, size), labeldistance=1.1)

plt.title('Percentage of Survived & Not Survived Passengers in Titanic', fontsize = 20)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.legend()
plt.show()


# In[13]:


size = list(numeric_df['Pclass'].value_counts())
labels = ['3rd Class', '1st Class', '2nd Class']
colors = ['aqua', 'blue', 'deepskyblue']
explode = [0, 0.1, 0.1]

def func(pct, allvals):
    absolute = int(round(pct/100*np.sum(allvals)))
    return "{:.1f}%\n({:d} Passengers)".format(pct, absolute)

plt.subplots(figsize=(8,8))
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, 
        autopct = lambda pct: func(pct, size), labeldistance=1.1)

plt.title('Percentage of Passengers \nfor different Fare classes in Titanic', fontsize = 20)
plt.legend()
plt.show()


'''
Passenegers Of Pclass 1 has a very high priority to survive.
The number of Passengers in Pclass 3 were a lot higher than Pclass 1 and Pclass 2, but still the number of survival from pclass 3 is low compare to them.
Pclass 1 %survived is around 63%, for Pclass2 is around 48%, and Pclass3 survived is around 25%
We saw that Sex and Class is important on the survive.So,Lets check survival rate with Sex and Pclass Together.'''


# In[14]:


color= ['Red', '#0da6ec']
def bar_plot(attribute, data, color, title, size, space, comparison = None, comparison_order=None):
    plt.figure(figsize=size)
    if comparison == None:
        ax = sns.countplot(x = attribute, data = data, palette=['Red', '#0da6ec'])
    else:
        ax = sns.countplot(x = attribute, hue = comparison, hue_order=comparison_order, data = data, palette=['Red', '#0da6ec'])
    total = len(data)
    
    for i in ax.patches:
        percentage = ' '*space + '{:.2f}%'.format((i.get_height()/total)*100)
        x = i.get_x()
        y = i.get_height()
        ax.annotate(percentage, (x,y))
    plt.title(title, size = 20)


# In[15]:


color= ['Red', '#0da6ec']
bar_plot('Pclass', numeric_df, 'cool', 
         "Percentage of Passengers \nfor different Fare classes \nbased on the Survival Status", 
         (10, 5), 3, 'Survived')

plt.legend(loc='upper left', labels=['Not Survived', 'Survived']);


# In[16]:


sns.catplot('Pclass','Survived',data=numeric_df, kind='point', color='deepskyblue');


# In[17]:


plt.figure(figsize=(10,5))
sns.kdeplot(data=numeric_df, x='Fare', hue='Survived', palette="seismic");
plt.title("Distribution of Fare based on the Survival Status \nof the Passengers in Titanic", fontsize = 20);

plt.legend(loc='upper right', labels=['Not Survived', 'Survived']);


# In[18]:


print(numeric_df.groupby('Pclass')['Fare'].mean())
numeric_df.groupby('Pclass')['Fare'].mean().plot.barh(color=[ 'blue', 'deepskyblue', 'aqua']);


# In[19]:


def horizontal_bar_plot(feature, dataframe, color, title, adjust, figsize, hue=None):
  # Create barplot 
  plt.figure(figsize=figsize)

  if hue == None:
    ax = sns.countplot(y=feature, data=dataframe, palette=[ "midnightblue", 'blue', 'deepskyblue', 'aqua',"darkcyan","lawngreen","coral"])
  else:
    ax = sns.countplot(y=feature, data=dataframe, palette=color, hue=hue)

  # Annotate every single Bar with its value, based on it's width           
  for p in ax.patches:
      width = p.get_width()
      plt.text(p.get_width()+adjust[0], p.get_y()+adjust[1]*p.get_height(),
              '{} Passesngers\n[{:.2f}%]'.format(int(width), width*100/train_df[feature].shape[0]),
              ha='center', va='center')
      
  plt.title(title, fontsize=23);


# In[20]:


horizontal_bar_plot('Parch', numeric_df, 'cool', 
                    "Percentage of Passengers \nwith different numbers of parents/children \naboard the Titanic",
                    (63, 0.55), (10, 6))


# In[21]:


bar_plot('Parch', numeric_df, ['Red', '#0da6ec'],
         "Percentage of Passengers with different \nnumbers of parents/children aboard the Titanic\nbased on the Survival Status",
         (12, 6), 1, 'Survived')

plt.legend(loc='upper right', labels=['Not Survived', 'Survived']);


# In[22]:


sns.catplot('Parch','Survived',data=numeric_df, kind='point', palette='tab10');


# In[23]:


def Feature_Bin(attribute, data, comparison = None):
    
    new_df = data.copy()
    
    intervals = [0, 2, 4, 9, 15, 21, 44, 60, 80]
    labels = ['Infant', 'Toddler', 'Child', 'Teenager', 'Adult', 'Mid-Age', 'Middle Senior', 'Old-Age']

    a = 'Different {} Grouped_Value'.format(attribute)
    new_df[a] = pd.cut(x = new_df[attribute], bins = intervals, labels = labels, include_lowest=True)
    

    total = len(new_df[a])
    
    if comparison == None:
        plt.figure(figsize=(10,8))
        ax = sns.countplot(x = a, data = new_df, palette="crest");
        for i in ax.patches:
          percentage = '  {:.2f}%'.format((i.get_height()/total)*100)
          x=i.get_x() 
          y=i.get_height()
          ax.annotate(percentage, (x,y))
        
        plt.title('Percentage of Passengers \nfor different Age-Group in Titanic', size=20)

    else:
        # Create a subplot
        f, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Create Percentage Plot
        ax = sns.countplot(x = a, data = new_df, hue = comparison, palette='Blues', ax=axes[0]);
        ax.set_title("Percentage of Passengers of different Age-Group \nbased on the Survival Status in Titanic").set_size(20)
        ax.legend(loc='upper right', labels=['Not Survived', 'Survived']);

        for i in ax.patches:
          percentage = '{:.2f}%'.format((i.get_height()/total)*100)
          x=i.get_x() 
          y=i.get_height()
          ax.annotate(percentage, (x,y))


        # Calculate Survival Rate
        percentage_value = [round(j, 2) for j in np.array([i.get_height()*100 for i in ax.patches])/total]
        non_survival_percentage_value = percentage_value[:8]
        survival_percentage_value = percentage_value[8:]
        survival_rate = [round(survival_percentage_value[i]*100/(survival_percentage_value[i] + non_survival_percentage_value[i]), 2) for i in range(8)]
        survival_rate_df = pd.DataFrame({'Age-Group':labels, 'Survival_Rate':survival_rate})

        # Create barplot of Survival Rate
        ax_one = sns.barplot(y='Survival_Rate', x='Age-Group', data=survival_rate_df, ax=axes[1], palette='crest')
        ax_one.set_title('Survival Rate of different Age-Group in Titanic').set_size(20)
        for i in ax_one.patches:
          percentage = '  {}%'.format(i.get_height())
          x=i.get_x() 
          y=i.get_height()
          ax_one.annotate(percentage, (x,y))


# In[24]:


Feature_Bin('Age', numeric_df) 


# In[25]:


Feature_Bin('Age', numeric_df, 'Survived')


# In[26]:


horizontal_bar_plot('SibSp', numeric_df, 'Set1', 
                    "Percentage of Passengers \nwith different numbers of siblings/spouses \naboard the Titanic",
                    (60, 0.55), (10, 6))


# In[27]:


bar_plot('SibSp', numeric_df, ['Red', '#0da6ec'],
         "Percentage of Passengers with different \nnumbers of siblings/spouses aboard the Titanic\nbased on the Survival Status",
         (12, 6), 1, 'Survived')

plt.legend(loc='upper right', labels=['Not Survived', 'Survived']);


# In[28]:


# create a list of all categorical features
categorical_features = train_df.select_dtypes(include=['object']).columns
categorical_features


# In[29]:


# create the dataframe of all categorical features
categorical_df = train_df[categorical_features]


# In[30]:


size = list(categorical_df['Embarked'].value_counts())
labels = ['Southampton', 'Cherbourg', 'Queenstown']
colors = ['#c4f8fe', '#ffadad', '#d5c4fe']

def func(pct, allvals):
    absolute = int(round(pct/100*np.sum(allvals)))
    return "{:.1f}%\n({:d} \nPassengers)".format(pct, absolute)


fig1, ax = plt.subplots(figsize=(8,8))
patches, texts, autotexts = ax.pie(size, labels = labels, colors = colors, shadow = True, 
                                    autopct = lambda pct: func(pct, size), labeldistance=0.9,
                                    startangle=180, counterclock=False, rotatelabels=True,)

plt.setp(texts, rotation_mode="anchor", ha="center", va="center")
for t, at in zip(texts, autotexts):
    rot = t.get_rotation()
    t.set_rotation(rot + 90 + (1 - rot // 180) * 180)
    at.set_rotation(t.get_rotation())


# draw circle
centre_circle = plt.Circle((0, 0), 0.45, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)


plt.title('Percentage of Passengers embarked from different Ports', fontsize = 20)
plt.legend()
plt.show()


# In[31]:


group_names = ['Southampton', 'Cherbourg', 'Queenstown']
group_size = list(categorical_df['Embarked'].value_counts())

subgroup_names = ['Not Survived', 'Survived']*3
subgroup_size = list(train_df.groupby(by='Embarked')['Survived'].value_counts())

# Create colors
a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

fig, ax = plt.subplots(figsize=(7,10))
ax.axis('equal')

plt.title('Percentage of Passengers embarked from different Ports \nbased on the Survival Status', fontsize = 20)

# First Ring (outside)
plt.pie(group_size, radius=1.5, labels=group_names, 
        colors=[a(0.8), b(0.8), c(0.8)], startangle=90)

# Second Ring (Inside)
plt.pie(subgroup_size, radius=1.3-0.3, 
                  labels=subgroup_names, labeldistance=1, 
                  colors=[b(0.2), b(0.5), c(0.2), c(0.5), a(0.2), a(0.5)], startangle=-9,
                  autopct = "%.2f%%", rotatelabels=True, pctdistance=0.75)
# draw circle
centre_circle = plt.Circle((0, 0), 0.5, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

plt.show()


# In[32]:


sns.catplot('Embarked','Survived',data=train_df, kind='point', palette='winter');


# In[33]:


horizontal_bar_plot('Sex', train_df, ['Red', '#0da6ec'], 
                    'Percentage of Male & Female Passengers \nin Titanic, based on the Survival Status',
                    (-35, 0.55), (10, 6), 'Survived')

plt.legend(loc='lower right', labels=['Not Survived', 'Survived']);


'''The number of men on the ship is lot more than the number of women.
But, the number of women saved is almost twice the number of males saved.
The survival rates for a women on the ship is around 75% while that for men in around 18-19%.
This looks very important feature for prediction the Survived people'''


# In[34]:


sns.catplot('Sex','Survived',data=train_df, kind='point', palette=['blue', "deeppink"]);


# In[35]:


sns.catplot('Pclass','Survived',hue='Sex',data=train_df, kind='point', palette=['blue', "deeppink"]);


# In[36]:


sns.catplot("Embarked", "Survived", hue="Sex", data=train_df, kind="point", palette=['blue', "deeppink"]);


# In[37]:


sns.catplot("Embarked", 'Survived', hue="Pclass", col="Sex", data=train_df, kind="point", palette=[ 'blue', 'green', 'red']);


# In[38]:


fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121)
g = sns.pointplot("Parch", 'Survived', hue="Sex", data=train_df, kind="point", palette=['blue', "deeppink"], ax=ax1);

ax2 = fig.add_subplot(122)
sns.pointplot("SibSp", 'Survived', hue="Sex", data=train_df, kind="point", palette=['blue', "deeppink"], ax=ax2);

plt.tight_layout()


# In[39]:


generations = [5, 10, 20, 30, 40, 50, 60, 70 , 80]
sns.lmplot("Age", "Survived",
            hue="Pclass", col="Sex",
            data=train_df,
            palette=["crimson","lime","royalblue"], x_bins=generations);


# In[43]:


horizontal_bar_plot('Sex', categorical_df, 
                    'Percentage of Male & Female \nPassengers in Titanic',
                    (-102, 0.55), (10, 6))


# In[48]:


# Group the data by 'Sex'
grouped_df = categorical_df.groupby('Sex').size()

fig, ax = plt.subplots()
ax.barh(grouped_df.index, grouped_df.values, color=['pink', 'blue'])

# Calculate percentage for each category
percentage_df = (grouped_df / grouped_df.sum()) * 100

# Add percentage data above the bars
for i, v in enumerate(percentage_df):
    plt.text(grouped_df.values[i] + 1, i, str(round(v, 2)) + '%')

# Set the title and axis labels
plt.title('Percentage of Male & Female \nPassengers in Titanic')
plt.xlabel('Count')
plt.ylabel('Sex')
# Remove the box around the chart
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.show()




# In[49]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.swarmplot("Pclass","Age", hue="Survived", data=train_df,split=True,ax=ax[0],palette='Set2')
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.swarmplot("Sex","Age", hue="Survived", data=train_df,split=True,ax=ax[1],palette='Set2')
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[51]:


f,ax=plt.subplots(1,2,figsize=(18,8))

custom_colors = ['Red', '#0da6ec']  # custom color codes

sns.swarmplot("Pclass","Age", hue="Survived", data=train_df, split=True, ax=ax[0], palette=custom_colors)
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.swarmplot("Sex","Age", hue="Survived", data=train_df, split=True, ax=ax[1], palette=custom_colors)
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))

plt.show()


# In[54]:


# Define a custom color palette
custom_palette = {"male": "blue", "female": "pink"}

# Plot the data with custom colors
sns.catplot(x="Age", y="Survived",
            hue="Sex", row="Pclass",
            data=train_df,
            orient="h", aspect=2, palette=custom_palette,
            kind="violin", dodge=True, cut=0, bw=.2)

