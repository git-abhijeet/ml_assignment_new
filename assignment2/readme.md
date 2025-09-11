# Tokyo 2021 Olympics Dataset Analysis - README

## 📊 Dataset Overview

This repository contains the Tokyo 2021 Olympics dataset downloaded from Kaggle, consisting of 5 Excel files with detailed information about athletes, medals, teams, coaches, and gender participation. The analysis assignment requires exploring this data using Pandas to generate 10 valuable insights.

## 📁 Dataset Files

### 1. Athletes.xlsx

-   **File Size**: 316KB
-   **Shape**: 11,085 rows × 3 columns
-   **Columns**:
    -   `Name`: Athlete's full name (string)
    -   `NOC`: National Olympic Committee code (string, e.g., "Norway", "Spain")
    -   `Discipline`: Sport/discipline name (string, e.g., "Cycling Road", "Artistic Gymnastics")
-   **Description**: Contains basic information for all participating athletes. Each row represents one athlete.
-   **Key Notes**:
    -   No individual medal information (medals are aggregated by country in Medals.xlsx)
    -   No physical attributes (height, weight, age) available
    -   Useful for participation counts and athlete distribution analysis
-   **Sample Data**:
    ```
                  Name     NOC           Discipline
    0    AALERUD Katrine  Norway         Cycling Road
    1        ABAD Nestor   Spain  Artistic Gymnastics
    2  ABAGNALE Giovanni   Italy               Rowing
    ```

### 2. Coaches.xlsx

-   **File Size**: 15KB
-   **Shape**: 394 rows × 4 columns
-   **Columns**:
    -   `Name`: Coach's full name (string)
    -   `NOC`: National Olympic Committee code (string)
    -   `Discipline`: Sport/discipline name (string)
    -   `Event`: Specific event (string, many NaN values)
-   **Description**: Information about team coaches. Each row represents one coach.
-   **Key Notes**:
    -   Many entries have NaN in the Event column
    -   Primarily useful for coach distribution analysis
    -   Can be used to understand coaching support by country/discipline
-   **Sample Data**:
    ```
                Name            NOC  Discipline Event
    0  ABDELMAGID Wael          Egypt    Football   NaN
    1        ABE Junya          Japan  Volleyball   NaN
    ```

### 3. EntriesGender.xlsx

-   **File Size**: 9.7KB
-   **Shape**: 46 rows × 4 columns
-   **Columns**:
    -   `Discipline`: Sport/discipline name (string)
    -   `Female`: Number of female participants (integer)
    -   `Male`: Number of male participants (integer)
    -   `Total`: Total participants (Female + Male) (integer)
-   **Description**: Gender participation breakdown for each discipline.
-   **Key Notes**:
    -   Perfect for gender equality analysis
    -   Some disciplines are gender-specific (e.g., Artistic Swimming has 0 males)
    -   Useful for understanding participation ratios
-   **Sample Data**:
    ```
              Discipline  Female  Male  Total
    0       3x3 Basketball      32    32     64
    1              Archery      64    64    128
    ```

### 4. Medals.xlsx

-   **File Size**: 6.9KB
-   **Shape**: 93 rows × 7 columns
-   **Columns**:
    -   `Rank`: Overall ranking by total medals (integer)
    -   `Team/NOC`: Country/National Olympic Committee name (string)
    -   `Gold`: Number of gold medals (integer)
    -   `Silver`: Number of silver medals (integer)
    -   `Bronze`: Number of bronze medals (integer)
    -   `Total`: Total medals (Gold + Silver + Bronze) (integer)
    -   `Rank by Total`: Ranking by total medals (integer)
-   **Description**: Medal tallies aggregated by country/NOC.
-   **Key Notes**:
    -   No individual athlete medal details
    -   Rankings are provided (Rank and Rank by Total)
    -   Essential for country-level performance analysis
-   **Sample Data**:
    ```
       Rank                    Team/NOC  Gold  Silver  Bronze  Total  Rank by Total
    0     1    United States of America    39      41      33    113              1
    1     2  People's Republic of China    38      32      18     88              2
    ```

### 5. Teams.xlsx

-   **File Size**: 25KB
-   **Shape**: 743 rows × 4 columns
-   **Columns**:
    -   `Name`: Team name (string, often the country name)
    -   `Discipline`: Sport/discipline name (string)
    -   `NOC`: National Olympic Committee code (string)
    -   `Event`: Gender category (string: "Men", "Women", or specific event)
-   **Description**: Information about participating teams.
-   **Key Notes**:
    -   Useful for team sport analysis
    -   Event column indicates gender (Men/Women)
    -   Can help distinguish team vs. individual sports
-   **Sample Data**:
    ```
        Name      Discipline                         NOC  Event
    0  Belgium  3x3 Basketball                     Belgium    Men
    1    China  3x3 Basketball  People's Republic of China    Men
    ```

## 🔗 Data Relationships

-   **Primary Key Connections**:
    -   `NOC` appears in Athletes.xlsx, Coaches.xlsx, Medals.xlsx, Teams.xlsx
    -   `Discipline` appears in Athletes.xlsx, Coaches.xlsx, EntriesGender.xlsx, Teams.xlsx
-   **Merging Strategy**:
    -   Athletes.xlsx + Medals.xlsx (on NOC) for athlete-medal analysis
    -   Athletes.xlsx + EntriesGender.xlsx (on Discipline) for gender participation
    -   Teams.xlsx + Medals.xlsx (on NOC) for team performance

## ⚠️ Data Limitations

-   No individual athlete medal details (only country aggregates)
-   No physical attributes (height, weight, age) for athletes
-   No performance metrics or timing data
-   Some columns have missing values (e.g., Event in Coaches.xlsx)

## 📋 Analysis Assignment Requirements

-   Generate 10 valuable insights using Pandas
-   Include data cleaning, transformation, and visualization
-   Focus on patterns in medal distribution, gender participation, sports analysis
-   Submit as Jupyter Notebook with code, outputs, and explanations

## 🛠️ Technical Setup

-   **Dependencies**: pandas, openpyxl, matplotlib, seaborn
-   **File Format**: Excel (.xlsx) files
-   **Python Version**: Compatible with Python 3.x

## 📈 Potential Insights

Based on available data, possible analysis areas:

1. Medal distribution by country
2. Gender participation trends
3. Sports with highest participation
4. Team vs. individual sport performance
5. Continental/regional comparisons
6. Discipline-level medal analysis
7. Coach distribution patterns
8. Participation ratios and equality metrics

---

_Last updated: September 4, 2025_
