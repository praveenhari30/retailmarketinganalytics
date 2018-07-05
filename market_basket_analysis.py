# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:47:05 2018

@author: Praveen
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

import os
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def cleanSalesTrx(path,filename):
    """
    Sales Transaction file has few junk characters at the start  of every row and empty column at the end.
    This function removes the junk characters and empty column and creates a cleaner version of the file.
    Return full file name to calling program.
    """
    messyFile = open(path+filename,encoding='latin_1')
    tidyFile = open(path+'SalesTrxCln.txt','a')
    for line in messyFile.readlines():
        tidyFile.write(line[4:]+'\n')
    tidyFile.close()
    print('Sales Transaction File is cleaned\n')
    return(tidyFile.name)

def extractSalesTrx(filename):
    """
    This function extracts the sales transaction data from cleaned Sales file to python
    It returns the sales transaction data frame
    """
    salesCols = ['StoreNum','Register','TransNum','TransDate','TransTime','BusDate','UPC','Item_Id','DeptNum','ItemQuantity',
             'WeightAmt','SalesAmt','CostAmt','CashierNum','PriceType','ServiceType','TenderType','LoyaltyCardNumber']
    salesTrx = pd.read_csv(filename,sep='|',
                     header=None,
                     names=salesCols,nrows=1000,
                     converters={6: str},
                     parse_dates= [3,5])
    salesTrx.name = 'SalesTrx'
    print("Sales Transaction file is read\n")
    return(salesTrx)

def extractItemAttr(path,filename):
    """
    This function extracts the sales transaction data from Item Attribute file to python
    It returns the Item Atrribute data frame
    """
    item_attr = pd.read_csv(path + filename,sep = '|',skiprows=3,header = None, converters={0: str})
    col_names = ["UPC","ItemPosDes","ItemAttributeDes","ItemAttributeValue","AttributeStartDate","AttributeEndDate"]
    item_attr.columns = col_names
    item_attr['AttributeStartDate'] =  pd.to_datetime(item_attr['AttributeStartDate'], format='%Y-%M-%d')
    item_attr['AttributeEndDate'] =  pd.to_datetime(item_attr['AttributeEndDate'], format='%Y-%m-%d')
    print("Item attribute data frame created\n")
    return (item_attr)


def extractCustomer(path,filename):
    """
    This function extracts the sales transaction data from Customer file to python
    It returns the Customer data frame
    """
    cust_list = pd.read_csv(path + filename,sep = '|',header = None,encoding='latin_1')
    col_names = ["LoyaltyCardNum","HouseholdNum","MemberFavStore","City","State","ZipCode","ec"]
    cust_list.columns = col_names
    cust_list = cust_list.drop(['ec'],axis = 1)
    cust_list['LoyaltyCardNum'] = cust_list['LoyaltyCardNum'].fillna(-999).astype(int)
    cust_list['HouseholdNum'] = cust_list['HouseholdNum'].fillna(-999).astype(int)
    cust_list['MemberFavStore'] = cust_list['MemberFavStore'].fillna(-999).astype(int)
    print('Customer data frame created\n')
    return (cust_list)

def extractItemList(path,filename):
    item_list = pd.read_csv(path+filename,sep = '|',header = None,encoding='latin1', converters={0: str})
    col_names = ["UPC","Item_Id","Status","LongDes","ShortDes","ClassCode","ClassDes","CategoryCode","CategoryDes",
             "FamilyCode","FamilyDes","DepartmentCode","StoreBrand","ExtraDes","ec"]
    item_list.columns = col_names
    item_list = item_list.drop(['ec'],axis = 1)
    print('Item list data frame created\n')
    return(item_list)

def extractStoreInfo(path,filename):
    store_info = pd.read_csv(path+filename,sep = '|',header = 1 ,encoding='latin1')
    col_names = ["store_nbr", "store_nm", "actv_rec_ind", "store_addr_line_1", "store_city_nm",
                 "store_state_prv_Cd", "store_post_cd", "store_sq_feet", "store_rgn_desc", "store_clst_desc","ec"]
    store_info.columns = col_names
    store_info = store_info.drop(['ec'],axis = 1)
    store_info['store_nbr'] = store_info['store_nbr'].str.extract('(\d+)')
    store_info['store_nbr'] = store_info['store_nbr'].fillna(-999).astype(int)
    store_info['store_post_cd'] = store_info['store_post_cd'].fillna(-999).astype(int)
    store_info['store_sq_feet'] = store_info['store_sq_feet'].fillna(-999).astype(int)
    print('Store Info data frame created\n')
    return(store_info)

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def storewise(iteminsalesDF,storenums,sup,lift,conf):
    basket={}

    for i in storenums:
        basket["{0}".format(i)] = (iteminsalesDF[iteminsalesDF['StoreNum']==i].groupby(['TransNum','LongDes'])['ItemQuantity'].sum().unstack().reset_index().fillna(0).set_index('TransNum'))

    basket_sets={}

    for j in basket.keys():
        basket_sets[j.format(j)] = basket.get(j).applymap(encode_units)

    frequent_itemsets={}
    for k in basket_sets.keys():
        frequent_itemsets[k.format(k)] = apriori(basket_sets.get(k), min_support=sup, use_colnames=True)

    rules={}
    for l in frequent_itemsets.keys():
        if(not(frequent_itemsets.get(l).empty) or frequent_itemsets.get(l) is None):
            rules[l.format(l)] = association_rules(frequent_itemsets.get(l), metric="lift", min_threshold=1)

    rulesfit={}
    for m in rules.keys():
        if(not(frequent_itemsets.get(m).empty) or frequent_itemsets.get(m) is None):
            if (rules.get(m)['lift'].all() >= lift) & (rules.get(m)['confidence'].all() >= conf):
                rulesfit[m.format(m)] = rules.get(m)

    for n in rulesfit.keys():
        if(not(rulesfit.get(n).empty) or rulesfit.get(n) is None):
            rulesfit.get(n).to_excel('C:/Users/Praveen/Documents/MSDS/ISQS6347-datamining/project/apriorirules/stores/rules'+n+'.xlsx')
    print('Store wise Market Basket analysis completed\n')


def customerwise(custiteminsalesDF,custnums,sup,lift,conf):
    basket={}
    for i in custnums:
        basket["{0}".format(i)] = (iteminsalesDF[iteminsalesDF['LoyaltyCardNumber']==i].groupby(['TransNum','LongDes'])['ItemQuantity'].sum().unstack().reset_index().fillna(0).set_index('TransNum'))

    basket_sets={}

    for j in basket.keys():
        basket_sets[j.format(j)] = basket.get(j).applymap(encode_units)

    frequent_itemsets={}
    for k in basket_sets.keys():
        frequent_itemsets[k.format(k)] = apriori(basket_sets.get(k), min_support=sup, use_colnames=True)

    rules={}
    for l in frequent_itemsets.keys():
        if(not(frequent_itemsets.get(l).empty) or frequent_itemsets.get(l) is None):
            rules[l.format(l)] = association_rules(frequent_itemsets.get(l), metric="lift", min_threshold=1)

    rulesfit={}
    for m in rules.keys():
        if(not(frequent_itemsets.get(m).empty) or frequent_itemsets.get(m) is None):
            if (rules.get(m)['lift'].all() >= lift) & (rules.get(m)['confidence'].all() >= conf):
                rulesfit[m.format(m)] = rules.get(m)

    for n in rulesfit.keys():
        if(not(rulesfit.get(n).empty) or rulesfit.get(n) is None):
            rulesfit.get(n).to_excel('C:/Users/Praveen/Documents/MSDS/ISQS6347-datamining/project/apriorirules/customer/rules'+n+'.xlsx')
    print('Customer wise Market Basket analysis completed\n')

if __name__ == '__main__':

    PATH = 'C:\\Users\\Praveen\\Documents\\MSDS\\ISQS6339-BusinessIntelligence\\myproject\\data_files\\'
    SalesFile = 'sls_dtl.txt'
    ItemAttrFile = 'Item_Attr.txt'
    CustFile = 'customer_List.txt'
    ItemListFile = 'Item_List.txt'
    StoreFile = 'store_list.txt'

#Data Preparation

#cleaning the sales transaction file
    SalesTrxClean = cleanSalesTrx(PATH,SalesFile)
#creating sales transaction data frame
    salesDF = extractSalesTrx(SalesTrxClean)
#creating item attribute data frame
    itemAttrDF = extractItemAttr(PATH,ItemAttrFile)
#creating customer data frame
    custDF = extractCustomer(PATH,CustFile)
#creating item list data frame
    itemListDF = extractItemList(PATH,ItemListFile)
#creating store data frame
    storeDF = extractStoreInfo(PATH,StoreFile)
#filtering NA values from the sales data frame
    salesDF['LoyaltyCardNumber'] = salesDF['LoyaltyCardNumber'].fillna(-999).astype(int)

#filtering items in the item list that have a sale record
    iteminsalesDF = pd.merge(salesDF, itemListDF[['LongDes','Item_Id']], left_on='Item_Id', right_on='Item_Id',how = 'left').drop_duplicates(keep='first')

    iteminsalesDF['TransNum'] = iteminsalesDF['TransNum'].astype('str')
    salesDF.dropna(axis=0, subset=['TransNum'], inplace=True)

#Market Basket Analysis
#counting the number of stores involved in the sales transactions
    storenums = salesDF['StoreNum'].unique()
    plt.hist(iteminsalesDF['LongDes'],label=None,bottom=None)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.set_title('Frequency plot of Items in Stores')
    plt.show()

#Market basket analysis for each store
    storewise(iteminsalesDF,storenums,sup=0.05,lift =0.85,conf =0.25)

#counting the number of customers involved in the sales transactions with a loyalty card number
    custiteminsalesDF = iteminsalesDF[iteminsalesDF['LoyaltyCardNumber']!=-999]
    custnums = custiteminsalesDF['LoyaltyCardNumber'].unique()

#Market basket analysis for each  customer
    customerwise(custiteminsalesDF,custnums,sup=0.05,lift=0.85,conf=0.25)

#Dimensional analysis of sales

#Transaction volume count per customer per transaction
    custinstorefreq=custiteminsalesDF.groupby(['StoreNum','LoyaltyCardNumber','TransNum']).size().reset_index(name='Txn Vol per cust per txn')
    custinstorefreq.sort_values(by = 'Txn Vol per cust per txn', ascending=False)

#frequency of visit of each customer
    custinstore=iteminsalesDF.groupby(['StoreNum','LoyaltyCardNumber']).size().reset_index(name='Cust Freq')
    custinstore.sort_values(by = 'Cust Freq', ascending=False)

#number of loyalty card holding customers per store
    custcountinstore=custiteminsalesDF.groupby(['StoreNum']).LoyaltyCardNumber.nunique().reset_index(name='Loyal Cust')
    custcountinstore.sort_values(by = 'Loyal Cust', ascending=False)
    plt.bar(custcountinstore['StoreNum'], height = custcountinstore['Loyal Cust'])
    frame2 = plt.gca()
    frame2.set_title('Stores with Loyal Customers')
    plt.show()

#transaction count per store
    transinstore=iteminsalesDF.groupby(['StoreNum'])['TransNum'].nunique().reset_index(name='Txn Vol per Store')
    transinstore.sort_values(by = 'Txn Vol per Store', ascending=False)
    plt.bar(transinstore['StoreNum'], height = transinstore['Txn Vol per Store'])
    frame3 = plt.gca()
    frame3.set_title('Transaction count per store')
    plt.show()

#High net worth customers
    custnetsale=custiteminsalesDF.groupby(['LoyaltyCardNumber'])['SalesAmt'].sum().reset_index(name='Net Sale')
    custnetsale.sort_values(by = 'Net Sale', ascending=False)

#Stores with high networth customers
    custnetsale=custiteminsalesDF.groupby(['LoyaltyCardNumber','StoreNum'])['SalesAmt'].sum().reset_index(name='Net Sale')
    custnetsale.sort_values(by = 'Net Sale', ascending=False)

#Stores sorted by net sales
    storenetsaledf = pd.DataFrame(custnetsale)
    storenetsale = storenetsaledf.groupby(['StoreNum'])['Net Sale'].sum().sort_values(ascending = False).reset_index(name='Store Net Sale')
    plt.bar(storenetsale['StoreNum'], height = storenetsale['Store Net Sale'])
    frame4 = plt.gca()
    frame4.set_title('Net Sale per store')
    plt.show()

#transaction time for each store
    iteminsalesDF['TransTime'] = pd.to_datetime(iteminsalesDF['TransTime'], format='%H:%M:%S').dt.time
    iteminsalesDF['TransHour'] = pd.to_datetime(iteminsalesDF['TransTime'], format='%H:%M:%S').dt.hour
    storesaletime=iteminsalesDF.groupby(['StoreNum','TransHour'])['TransHour'].size().reset_index(name='Frequent TransHours')
    storesaletime.sort_values(by = 'Frequent TransHours', ascending=False)