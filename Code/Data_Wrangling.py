import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os
sns.set()


# reading the files
# df_read = pd.read_csv("../Data/Global-Superstore.csv")

# Change encoding type due to error when importing the file with 'utf-8'
df_read = pd.read_csv("../Data/Global-Superstore.csv", encoding= "latin-1")

# displaying all rows and looking to few rows
pd.set_option("display.max_columns", None)
df_read.head()

# looking into table summary
df_read.describe(include= "all").fillna("-")

# Checking Null
df_read.isna().sum()

# Checking duplicated values
df_read.duplicated().sum()

# Checking data types
df_read.info()


# Dropping unwanted coumns
df_read.columns

data = df_read[[
    #'Row ID', 'Order ID', 
    'Order Date', 
    # 'Ship Date', 'Ship Mode',
    # 'Customer ID', 'Customer Name', 'Segment', 'City', 'State', 
    'Country',
    # 'Postal Code', 
    'Market', 
    # 'Region', 'Product ID', 
    'Category',
    'Sub-Category', 'Product Name', 'Sales', 'Quantity', 
    # 'Discount',
    # 'Profit', 'Shipping Cost', 'Order Priority'
]]


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Function to use in exploring the data frame "aggregation and grouping"
def group_agg_type(
        agg_type, dataframe, group_on, agg_on, method= None, col_name = None, sort_column= None, ascending= None
        ):
    
    if agg_type == "single":

        # Function for single aggregation
        def agg_single(dataframe, group_on, agg_on, method, col_name):
            group_by_result = dataframe.groupby(group_on)

            agg_method_func = getattr(group_by_result[agg_on], method)
            result = agg_method_func().reset_index(name= col_name)

            return result
        
        return agg_single(dataframe, group_on, agg_on, method, col_name)
        
    elif agg_type == "multi":

        # Function for multi aggregation
        def agg_multi(dataframe, group_on, agg_on, sort_column, ascending):
            group_by_result = dataframe.groupby(group_on)

            aggregated_columns = []
            for col, agg_method, col_name in agg_on:

                agg_method_func = getattr(group_by_result[col], agg_method)
                result = agg_method_func().reset_index(name=col_name)

                aggregated_columns.append(result)

            final_result = (pd.concat(aggregated_columns, axis=1).round(2).reset_index(drop= True))
            final_result = final_result.loc[:, ~final_result.columns.duplicated(keep= "first")]

            return final_result.sort_values(by = sort_column, ascending= ascending)

        return agg_multi(dataframe, group_on, agg_on, sort_column, ascending)
    
    else:
        raise ValueError("Check Parameters!")

### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Function to check columns with hierarchy
def check_hierarchy(df_check, group_col, check_col):

    # group data and count unique values
    col_hierarchy = group_agg_type(
        agg_type= "single", dataframe= df_check, group_on= group_col,
        agg_on= check_col, method= "nunique", col_name= f"{check_col}_Count"
    )

    # returns when values from both condition are equal
    if col_hierarchy[f"{check_col}_Count"].sum() == df_check[check_col].nunique():
        print("No, Inconsistency")

    # returns when values from both are not equal
    else:
        print(f"Number of Inconsistencies in {check_col}: {col_hierarchy[f'{check_col}_Count'].sum() - df_check[check_col].nunique()}")


# Function to extract the value that produce anomaly in data hierarchy
def extract_anomaly(extract_in, df_check, group_col, anomaly_on):
    
    # add unique values column using index value
    df_check["Row_no"] = df_check.index
    
    # group data and count the unique values using 'size' method
    col_hierarchy =  group_agg_type(
        agg_type= "single", dataframe= df_check[["Row_no", group_col, anomaly_on]],
        group_on= [group_col, anomaly_on] , agg_on= "Row_no", method= "size", col_name= "Row_Count")

    # group data and return values with more than 1 count
    col_anomaly =  group_agg_type(
        agg_type= "single", dataframe= col_hierarchy, group_on= anomaly_on,
        agg_on= anomaly_on, method= "size", col_name= "Count").loc[lambda x: x["Count"]> 1]
    
    # stores extracted values and drop the temporary "Row_no" column
    col_result = col_anomaly
    df_check.drop(columns = "Row_no", inplace= True)

    # returns the first group operation
    if extract_in == "column":

        return col_result
    
    # returns the second group operation
    elif extract_in == "hierarchy":

        hierarchy_anomaly = (col_hierarchy[col_hierarchy[anomaly_on].isin(col_anomaly[anomaly_on])])
        
        return hierarchy_anomaly
    
    else:
        raise ValueError("Check Parameters! If hierarchy columns doesn't have inconsistency, this function wouldn't work!")


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Checking Market & Country hierarchy
check_hierarchy(df_check= data, group_col= "Market", check_col= "Country")

# Extracting Inconsistent data in Country
extract_anomaly(extract_in= "column", df_check= data, group_col= "Market", anomaly_on= "Country")

# Extracting Inconsistent data hierarchy in the data frame
extract_anomaly(extract_in= "hierarchy", df_check= data, group_col= "Market", anomaly_on= "Country")

# Based on the result, Mongolia belongs to  APAC, while Austria is part of EMEA
# dictionary to map and replace the incorrect data
Correct_dict = {
    "APAC": ["Mongolia"],
    "EMEA": ["Austria"]
}

# Dataframe to hold the values from dictionary
Correct_market = pd.DataFrame.from_dict(Correct_dict, orient= "index").reset_index()
Correct_market.columns= ["Market", "Country"]

# mapping the right values
data["New Market"] = np.where(data["Country"].isin(Correct_market["Country"]),
                              data["Country"].map(Correct_market.set_index("Country")["Market"]),
                              data["Market"]
)

# Checking hierarchy with the 'New Market'
check_hierarchy(df_check= data, group_col= "New Market", check_col= "Country")

# dropping the old market column and renaming 'New Market' to 'Market'
data.drop(columns= "Market", inplace= True),
data.rename(columns= {"New Market" : "Market"}, inplace= True)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Checking Category & Sub-Category hierarchy
check_hierarchy(df_check= data, group_col= "Category", check_col= "Sub-Category")

# Checking Sub-Category & Product Name hierarchy
check_hierarchy(df_check= data, group_col= "Sub-Category", check_col= "Product Name")

# Extracting Inconsistent data in Country
extract_anomaly(extract_in= "column", df_check= data, group_col= "Sub-Category", anomaly_on= "Product Name")

# Extracting Inconsistent data hierarchy in the data frame
Inconsistent_Product = extract_anomaly(extract_in= "hierarchy", df_check= data, group_col= "Sub-Category", anomaly_on= "Product Name")


# Create a copy of the original DataFrame
updating_target = data.copy()

# Based on the result, product 'Staples' should logically be under Sub-Category "Fasteners"
# Adding the right Sub-Category and Category

# Values to use
new_sub_category = "Fasteners"
new_category = "Office Supplies"

# Create a boolean mask for products in Inconsistent_Product
target = updating_target["Product Name"].isin(Inconsistent_Product["Product Name"])

# Update Sub-category and Category based on the mask
updating_target["New Sub-Category"] = np.where(target, new_sub_category, updating_target["Sub-Category"])
updating_target["New Category"] = np.where(target, new_category, updating_target["Category"])

# Checking the result with the incosistent product
check_hierarchy(df_check= updating_target, group_col= "New Sub-Category", check_col= "Product Name")


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Adding date formatted column on the data frame, dropping columns applying on a copy dataframe
updating_target.insert(
    0 + 1, "Formatted Date",
    pd.to_datetime(updating_target["Order Date"])),
updating_target.insert(
    1 + 1, "Year",
    updating_target["Formatted Date"].dt.year),
updating_target.insert(
    2 + 1, "Month",
    updating_target["Formatted Date"].dt.month), 
updating_target.insert(
    3 + 1, "Quarter",
    updating_target["Formatted Date"].dt.quarter),
updating_target.drop(columns= ["Order Date", "Sub-Category","Category"], inplace= True)

# Re-organizing the data frame and renaming columns
updating_target.columns

data_cleaned = updating_target[
    ['Formatted Date', 'Year', 'Month', 'Quarter', 'Country', 'Market',
       'New Sub-Category', 'New Category', 'Product Name',
       'Sales', 'Quantity']
       ].rename(
    columns= {"New Sub-Category" : "Sub Category", "New Category" : "Category"}
    )


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Yearly Overview
# columns to aggregate, aggregations and column result names
agg_list_yearly = [
    ("Formatted Date", "count", "Total_Transaction"),
    ("Sales", "sum", "Total_Sales")
]

yearly_summary = (
    group_agg_type(agg_type= "multi", dataframe= data_cleaned, group_on= ["Year", "Month"],
                   agg_on= agg_list_yearly, sort_column= ["Year","Month"], ascending= False)
)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###

### Plotting Total Revenue by Year
yearly_rev_pivot = yearly_summary.pivot(index= "Month", columns= "Year", values= "Total_Sales")

yearly_rev_pivot.plot(marker= "o", figsize=(10, 6))
plt.title("Yearly Total Sales by Month", fontsize=18)
plt.xlabel("Month", fontsize= 15)
plt.ylabel("Total Revenue", fontsize= 15)

for month in [3.5, 6.5, 9.5]:
    plt.axvline(x=month - 0.5, color='gray', linestyle='--', linewidth=1)

# Show Full Months
plt.xticks(yearly_rev_pivot.index)
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###

### Plotting Total Transaction by Year
yearly_transact_pivot = yearly_summary.pivot(index= "Month", columns= "Year", values= "Total_Transaction")

yearly_transact_pivot.plot(marker= "o", figsize=(10,6))
plt.title("Yearly Total Transaction by Month", fontsize= 18)
plt.xlabel("Month", fontsize= 15)
plt.ylabel("Total Transaction", fontsize= 15)

for month in [3.5, 6.5, 9.5]:
    plt.axvline(x=month - 0.5, color='gray', linestyle='--', linewidth=1)

# Show Full Months
plt.xticks(yearly_rev_pivot.index)
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Market overview
# columns to aggregate, aggregations and column result names
agg_list_market =[
    ("Country", "nunique", "Total_Country"),
    ("Sales", "sum", "Total_Sales"),
    ("Sales", "mean", "Average_Sales"),
    ("Quantity", "sum", "Total_Orders"),
    ("Quantity", "mean", "Average_Orders")
]

# Result
Market_summary = (
    group_agg_type(agg_type= "multi", dataframe= data_cleaned, group_on= "Market", 
                   agg_on= agg_list_market, sort_column= "Total_Sales", ascending= False)
)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Function to alter values format
def format_millions(x, pos):
    return f'{x/1e6:.1f}-M'

def format_thousands(x, pos):
    return f'{x/1e3:.0f}-K'

# Variable millions formatter
format_million = FuncFormatter(format_millions)

# Variable thousand formatter
format_thousand = FuncFormatter(format_thousands)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


### Plotting Market "Revenue"
plt.bar(Market_summary["Market"] , Market_summary["Total_Sales"])

# Graph title and labels
plt.title("Total Revenue by Market", fontsize= 18)
plt.xlabel("Market", fontsize= 15)
plt.ylabel("Revenue", fontsize = 15)

# Applying modified format
plt.gca().yaxis.set_major_formatter(format_million)
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


### Plotting Market "Orders"
plt.bar(Market_summary["Market"] , Market_summary["Total_Orders"])

# Graph title and labels
plt.title("Total Orders by Market", fontsize= 18)
plt.xlabel("Market", fontsize= 15)
plt.ylabel("Orders", fontsize = 15)

# Applying modified format
plt.gca().yaxis.set_major_formatter(format_thousand)
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


### Plotting total Countries under their Market and Average sales
fig, ax1 = plt.subplots()
ax1.bar(Market_summary["Market"], Market_summary["Total_Country"])

# axis to share with the sub-plot
ax2 = ax1.twinx()
ax2.plot(Market_summary["Market"], Market_summary["Average_Sales"],
          color="red", linestyle= "dashed", marker= "o", label= "Average Sales")

# Graph title and labels
plt.title("Number of Country and Average Sales", fontsize= 18, y= 1.05)
ax1.set_xlabel("Market", fontsize=15)
ax1.set_ylabel("Count of Country", fontsize=15)

# Hiding sub-plot axis values
ax2.set_yticks([])
ax2.set_yticklabels([])

# Displaying the subplot value above its markers
for i, txt in enumerate(Market_summary["Average_Sales"]):
    ax2.annotate(f"{txt:.2f}", (i, txt), textcoords="offset points", xytext=(0, 8), ha='center')

# Adding sub-plot legend for clarification
ax2.legend(loc="upper right")
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


### Plotting total Countries under their Market and Average Orders
fig, ax1 = plt.subplots()
ax1.bar(Market_summary["Market"], Market_summary["Total_Country"])

# axis to share with the sub-plot
ax2 = ax1.twinx()
ax2.plot(Market_summary["Market"], Market_summary["Average_Orders"],
          color="red", linestyle= "dashed", marker= "o", label= "Average Orders")

# Graph title and labels
plt.title("Number of Country and Average Orders", fontsize= 18, y= 1.05)
ax1.set_xlabel("Market", fontsize=15)
ax1.set_ylabel("Count of Country", fontsize=15)

# Hiding sub-plot axis values
ax2.set_yticks([])
ax2.set_yticklabels([])

# Displaying the subplot value above its markers
for i, txt in enumerate(Market_summary["Average_Orders"]):
    ax2.annotate(f"{txt:.2f}", (i, txt), textcoords="offset points", xytext=(0, 8), ha='center')

# Adding sub-plot legend for clarification
ax2.legend(loc="upper right")
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# narrowing down to profitable markets based on the plot results
data_narrowed = data_cleaned[data_cleaned["Market"].isin(["APAC", "EU", "US", "LATAM"])].reset_index(drop= True)

# columns to aggregate, aggregations and column result names
agg_list_categ = [
    ("Product Name", "nunique", "Total_Products"),
    ("Sales", "sum", "Total_Sales"),
    ("Quantity", "sum", "Total_Orders")
]

# Result
Category_summary = (
    group_agg_type(
        agg_type= "multi", dataframe= data_narrowed,
        group_on= "Category", agg_on=agg_list_categ, sort_column="Total_Sales", ascending=False)
)

# Sub-Category Monthly Performance
# columns to aggregate, aggregations and column result names
agg_list_months_subcateg = [
    ("Product Name", "nunique", "Total_Products"),
    ("Sales", "sum", "Total_Sales"),
    ("Sales", "sum", "Average_Sales"),
    ("Quantity", "sum", "Total_Orders")
]

# Result
SubCateg_months_overview = (
    group_agg_type(
        agg_type= "multi", dataframe= data_narrowed, group_on= ["Month", "Category", "Sub Category"],
        agg_on= agg_list_months_subcateg, sort_column= "Total_Sales", ascending= False)
)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


## Plotting Sub-Category performance for Category "Technology"
tech_data = SubCateg_months_overview[SubCateg_months_overview["Category"] == "Technology"]

# Group data to be pivot
grouped_tech_data = (
    group_agg_type(
        agg_type= "single", dataframe= tech_data, group_on= ["Month", "Sub Category"],
        agg_on= "Total_Sales", method= "sum", col_name= "Total_Sales")
)

# Pivoted the data for plotting
pivot_tech_data = grouped_tech_data.pivot(index='Month', columns='Sub Category', values='Total_Sales')

# Plotting and setting title and labels
pivot_tech_data.plot(marker='o', figsize=(10, 6))
plt.title("Total Sales for Technology Subcategories by Month", fontsize=18)
plt.xlabel("Month", fontsize=15)
plt.ylabel("Total Sales", fontsize=15)

# Show Full Months
plt.xticks(pivot_tech_data.index)
plt.show()


# Extracting profitable Sub-Category under Technology based on the plot result
Tech_Selected = SubCateg_months_overview[
    (SubCateg_months_overview["Category"] == "Technology") & 
    (SubCateg_months_overview["Sub Category"].isin(["Phones","Copiers"]))
    ][["Category","Sub Category"]].drop_duplicates()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


## Plotting Sub-Category performance for Category "Furniture"
Furniture_data = SubCateg_months_overview[SubCateg_months_overview["Category"] == "Furniture"]

# Group_agg_one data to be pivot
grouped_furniture_data = Furniture_data.groupby(["Month", "Sub Category"])["Total_Sales"].sum().reset_index()

# Pivoted the data for plotting
pivot_tech_furniture = grouped_furniture_data.pivot(index='Month', columns='Sub Category', values='Total_Sales')

# Plotting and setting title and labels
pivot_tech_furniture.plot(marker='o', figsize=(10, 6))
plt.title("Total Sales for Furniture Subcategories by Month", fontsize=18)
plt.xlabel("Month", fontsize=15)
plt.ylabel("Total Sales", fontsize=15)

# Show Full Months
plt.xticks(pivot_tech_furniture.index)
plt.show()


# Extracting profitable Sub-Category under Furniture based on the plot result
Furniture_Selected = SubCateg_months_overview[
    (SubCateg_months_overview["Category"] == "Furniture") & 
    (SubCateg_months_overview["Sub Category"].isin(["Bookcases","Chairs"]))
    ][["Category","Sub Category"]].drop_duplicates()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


## Plotting Sub-Category performance for Category "Office Supplies"
OfficeSup_data = SubCateg_months_overview[SubCateg_months_overview["Category"] == "Office Supplies"]

# Group_agg_one data to be pivot
grouped_officeSup_data = (
    group_agg_type(
        agg_type= "single", dataframe= OfficeSup_data, group_on= ["Month", "Sub Category"],
        agg_on= "Total_Sales", method= "sum", col_name= "Total_Sales")
)

# Pivoted the data for plotting
pivot_officesup_furniture = grouped_officeSup_data.pivot(index='Month', columns='Sub Category', values='Total_Sales')

# Plotting and setting title and labels
pivot_officesup_furniture.plot(marker='o', figsize=(10, 6))
plt.title("Total Sales for Office Supplies Subcategories by Month", fontsize=18)
plt.xlabel("Month", fontsize=15)
plt.ylabel("Total Sales", fontsize=15)

# Show Full Months
plt.xticks(pivot_tech_furniture.index)
plt.show()


# Extracting profitable Sub-Category under Office Supplies based on the plot result
OfficeSup_Selected = SubCateg_months_overview[
    (SubCateg_months_overview["Category"] == "Office Supplies") & 
    (SubCateg_months_overview["Sub Category"].isin(["Appliances","Storage"]))
    ][["Category","Sub Category"]].drop_duplicates()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Combining Selected Sub-Categories
combined_selected = pd.concat([Tech_Selected, Furniture_Selected, OfficeSup_Selected], axis= 0, ignore_index= True)

# Filtering the dataframe with the selected Sub-Categories
Filtered_data = data_narrowed[data_narrowed["Sub Category"].isin(combined_selected["Sub Category"])]

# Checking the filtered data
Filtered_data[["Category", "Sub Category"]].drop_duplicates().reset_index(drop= True)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###
Filtered_data["Year"].unique()

# Function to extract top products for each selected Sub-category and present in years 2011-2014
def quarter_top_products(df, quarter, top):

    # filtering the specfied dataframe with the condition
    quarter_df = df[df["Year"].isin([2011, 2012, 2013, 2014]) & (df["Quarter"]== quarter)]

    # columns to aggregate, aggregations and column result names
    agg_list = [
        ("Sales", "sum", "Total_Sales"),
        ("Quantity", "sum", "Total_Quantity")
    ]

    # group data and applying aggregation methods
    grouped_quarter_df = group_agg_type(
        agg_type= "multi", dataframe= quarter_df, group_on= ["Quarter", "Category","Sub Category","Product Name"],
        agg_on= agg_list, sort_column= "Total_Sales", ascending= False)
    
    # Function to extract top n products
    def top_n(df_check, column, n= None):
        return df_check.nlargest(n, column)

    # extracting top products by applying the function
    top_n_products = (
        grouped_quarter_df.groupby(["Quarter", "Sub Category"], group_keys= False)
        .apply(lambda x: top_n(x, column= "Total_Sales", n= top)).reset_index(drop= True)
        )
    
    return top_n_products

# Extracting top 10 products for each quarter
top_products_quarters = pd.concat(
    [quarter_top_products(df = Filtered_data, quarter= 1, top= 10),
     quarter_top_products(df = Filtered_data, quarter= 2, top= 10),
     quarter_top_products(df = Filtered_data, quarter= 3, top= 10),
     quarter_top_products(df = Filtered_data, quarter= 4, top= 10)], 
     axis= 0, ignore_index= True).drop_duplicates()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


### Plotting Selected Products and Unselected Products Total Revenue
unselected_products = Filtered_data[~Filtered_data["Product Name"].isin(top_products_quarters["Product Name"])]
selected_products = Filtered_data[Filtered_data["Product Name"].isin(top_products_quarters["Product Name"])]

unselected_group_sales = group_agg_type(
    agg_type= "single", dataframe= unselected_products, group_on= "Sub Category", 
    agg_on= "Sales", method= "sum", col_name= "Total_Sales"
    )

selected_group_sales = group_agg_type(
    agg_type= "single", dataframe= selected_products, group_on= "Sub Category",
    agg_on= "Sales", method= "sum", col_name= "Total_Sales"
)

concat_products = pd.concat([selected_group_sales, unselected_group_sales], keys=["Selected Products", "Unselected Products"])
pivoted_products = concat_products.reset_index().pivot(index= "Sub Category", columns= "level_0", values= "Total_Sales")

colors = {"Selected Products" : "blue", "Unselected Products": "gray"}

ax = pivoted_products.plot(kind='bar', color=[colors[col] for col in pivoted_products.columns])
ax.legend(['Selected Products', 'Unselected Products'])

plt.xlabel('Sub Category')
plt.ylabel('Total Sales')
plt.title('Total Sales Comparison by Sub Categories')

plt.gca().yaxis.set_major_formatter(format_million)
plt.ylim(0, 1000000)
plt.show()


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Total count of selected products
selected_df_check = pd.DataFrame({ "Selection_Status" : ["Selected"]})

selected_df_check["Total_Products"] = (
    top_products_quarters["Product Name"].nunique()
    )

# total revenue of selected products
selected_df_check["Total_Revenue"] = (
    data_cleaned[data_cleaned["Product Name"].isin(top_products_quarters["Product Name"])]["Sales"].sum().round(2)
)


# total count of unselected products
unselected_df_check = pd.DataFrame({ "Selection_Status" : ["Unselected"]})

unselected_df_check["Total_Products"] = (
    data_cleaned[~data_cleaned["Product Name"].isin(top_products_quarters["Product Name"])]["Product Name"].nunique()
)

# total revenue for unselected products
unselected_df_check["Total_Revenue"]= (
    data_cleaned[~data_cleaned["Product Name"].isin(top_products_quarters["Product Name"])]["Sales"].sum().round(2)
)

# Combining selected & unselected
Sel_Unsel = pd.concat([selected_df_check, unselected_df_check],axis= 0, ignore_index= True)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


## Plotting total revenue and product count to compare selected & unselected products
bars = plt.bar(Sel_Unsel["Selection_Status"] , Sel_Unsel["Total_Revenue"],width= 0.7)

# Graph title and labels
plt.title("Selected vs Unselected Products", fontsize= 18)
plt.xlabel("Selection Status", fontsize= 15)
plt.ylabel("Revenue", fontsize = 15)

for bar, total_products in zip(bars, Sel_Unsel["Total_Products"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{total_products} - Total Products",
             ha= "center", va= "bottom", fontsize= 12, fontweight= "bold")

# Applying modified format
plt.gca().yaxis.set_major_formatter(format_million)
plt.ylim(0, 10500000)
plt.show()

# comparing revenue of selected vs grand total revenue
(selected_df_check["Total_Revenue"] / (unselected_df_check["Total_Revenue"] + selected_df_check["Total_Revenue"])).round(2) * 100


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Exporting data frames to make data model in excel for dynamic visualization
# Function for exporting
def export_file(dataframe, filename):
    csv_path = os.path.join("../Data_Modeling_(Tables)", filename)
    dataframe.to_csv(csv_path, index= False)


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# DIM_Products table
# Concatinating selecting products
product_table_selected = top_products_quarters[["Product Name", "Sub Category", "Category"]].drop_duplicates().reset_index(drop= True)

# Adding 'Flag' column
product_table_selected["Flag"] = "Selected"

# Extracting unselected products
product_table_unselected = (
    data_cleaned[["Product Name", "Sub Category", "Category"]]
    [~data_cleaned["Product Name"].isin(product_table_selected["Product Name"])].drop_duplicates().reset_index(drop= True)
)

# Adding 'Flag' column
product_table_unselected["Flag"] = "Unselected"

# Concatinating selected and unselected product tables
DIM_Product = pd.concat([product_table_selected, product_table_unselected], axis= 0 , ignore_index= True)

# Adding primary key "Product ID"
DIM_Product.insert(0, "Product ID", range(1, len(DIM_Product)+ 1))


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# DIM_location table
# Extracting unique country
DIM_location = data_cleaned[["Country","Market"]].drop_duplicates().reset_index(drop= True)

target_markets= data_narrowed[["Market"]].drop_duplicates()

# Looping to Add flag
for x in DIM_location.index:
    if DIM_location.loc[x, "Market"] in data_narrowed["Market"].values:
        DIM_location.at[x,"Flag"] = "Selected"
        
    else:
        DIM_location.at[x, "Flag"] = "Unselected"

# Adding primary key "Country ID"
DIM_location.insert(0, "Country ID", range(100, 100 + len(DIM_location)))


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# DIM_Date
# Identifying starting date and last date
data_cleaned["Formatted Date"].min()
data_cleaned["Formatted Date"].max()

start_date = "2011-01-01"
end_date = "2014-12-31"

# data range
date_range = pd.date_range(start= start_date, end= end_date, freq= "D")

# Dataframe
DIM_date = pd.DataFrame({"Date": date_range})

# Adding additional date columns
DIM_date["Year"] = DIM_date["Date"].dt.year
DIM_date["Month_No"] = DIM_date["Date"].dt.month
DIM_date["Month_Name"] = DIM_date["Date"].dt.month_name().str[:3]
DIM_date["Quarter"] = DIM_date["Date"].dt.quarter


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# FACT_table table
FACT_table = data_cleaned[[
    'Formatted Date', 
    # 'Year', 'Month', 'Quarter', 
    'Country', 
    # 'Market','Sub Category', 'Category', 
    'Product Name', 'Sales', 'Quantity']].rename(columns={"Formatted Date" : "Date"})

# Create a dictionary for mapping values
product_id_mapping = dict(zip(DIM_Product["Product Name"], DIM_Product["Product ID"]))
country_id_mapping = dict(zip(DIM_location["Country"], DIM_location["Country ID"]))
 
# Adding product ID to Fact table
FACT_table["Product ID"]= FACT_table["Product Name"].map(product_id_mapping).astype("Int64")

# Adding Country ID to Fact table
FACT_table["Country ID"] = FACT_table["Country"].map(country_id_mapping)

# Organizing columns and dropping columns
FACT_table.columns

FACT_table = FACT_table[[
    'Date', 'Product ID', 'Country ID',
    #'Country', 'Product Name', 
    'Sales', 'Quantity'
]]


### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ### ---- ###


# Exporting files into CSV's

# DIM_Product CSV
export_file(DIM_Product, "DIM_Product.csv")

# DIM_location CSV
export_file(DIM_location, "DIM_location.csv")

# DIM_date CSV
export_file(DIM_date, "DIM_date.csv")

# FACT_table CSV
export_file(FACT_table, "FACT_table.csv")

# Overview Comparison CSV
export_file(Sel_Unsel, "Overview.csv")

