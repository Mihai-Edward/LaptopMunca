from collections import Counter
from itertools import combinations
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories

class DataAnalysis:
    def __init__(self, draws, debug=False):
        """Initialize DataAnalysis with proper draw validation"""
        self.draws = []
        self.debug = debug
        
        if self.debug:
            print(f"\nDEBUG: Initializing DataAnalysis with {len(draws)} draws")
            print(f"DEBUG: First draw format: {draws[0] if draws else 'None'}")  # Add this line
        
        for draw_date, numbers in draws:
            try:
                # Ensure numbers is a list and contains valid numbers
                numbers = [n for n in numbers if isinstance(n, (int, float)) and 1 <= n <= 80]
                
                # Remove duplicates while preserving order
                valid_numbers = []
                seen = set()
                for num in numbers:
                    if num not in seen and len(valid_numbers) < 20:
                        valid_numbers.append(num)
                        seen.add(num)
                
                # Only add if we have exactly 20 valid numbers
                if len(valid_numbers) == 20:
                    self.draws.append((draw_date, valid_numbers))
                else:
                    print(f"DEBUG: Skipping draw {draw_date} - invalid number count: {len(valid_numbers)}")
                    
            except Exception as e:
                print(f"DEBUG: Error processing draw {draw_date}: {e}")
        
        if not self.draws:
            raise ValueError("No valid draws were processed!")
        
        # if self.debug:
        #     print(f"DEBUG: Successfully processed {len(self.draws)} valid draws")

    def count_frequency(self):
        """Count frequency of all numbers across all draws"""
        all_numbers = [number for draw in self.draws for number in draw[1]]
        frequency = Counter(all_numbers)
        return frequency

    def get_top_numbers(self, top_n=20):
        """Get top N most frequent numbers"""
        frequency = self.count_frequency()
        most_common_numbers = [number for number, count in frequency.most_common(top_n)]
        print(f"DEBUG: Retrieved top {len(most_common_numbers)} numbers")
        return most_common_numbers

    def suggest_numbers(self, top_n=20):
        """Suggest numbers based on frequency analysis"""
        return self.get_top_numbers(top_n)

    def find_common_pairs(self, top_n=10):
        """Find most common pairs of numbers"""
        pairs = Counter()
        for draw in self.draws:
            numbers = draw[1]
            pairs.update(combinations(sorted(numbers), 2))
        common_pairs = pairs.most_common(top_n)
        print(f"DEBUG: Found {len(common_pairs)} common pairs")
        return common_pairs

    def find_consecutive_numbers(self, top_n=10):
        """Find most common consecutive number pairs"""
        consecutive_pairs = Counter()
        for draw in self.draws:
            numbers = sorted(draw[1])
            for i in range(len(numbers) - 1):
                if numbers[i] + 1 == numbers[i + 1]:
                    consecutive_pairs.update([(numbers[i], numbers[i + 1])])
        return consecutive_pairs.most_common(top_n)

    def number_range_analysis(self):
        """Analyze number distribution across ranges"""
        ranges = {
            '1-20': 0,
            '21-40': 0,
            '41-60': 0,
            '61-80': 0
        }
        
        for draw in self.draws:
            for number in draw[1]:
                if 1 <= number <= 20:
                    ranges['1-20'] += 1
                elif 21 <= number <= 40:
                    ranges['21-40'] += 1
                elif 41 <= number <= 60:
                    ranges['41-60'] += 1
                elif 61 <= number <= 80:
                    ranges['61-80'] += 1
                    
        print(f"DEBUG: Range analysis completed. Distribution: {ranges}")
        return ranges

    def hot_and_cold_numbers(self, top_n=10):
        """Identify hot (frequent) and cold (rare) numbers"""
        frequency = self.count_frequency()
        # Get full sorted frequency list
        sorted_numbers = frequency.most_common()
        
        # Get hot numbers (most frequent)
        hot_numbers = sorted_numbers[:top_n]
        
        # Get cold numbers (least frequent)
        cold_numbers = sorted_numbers[-top_n:]
        
        return hot_numbers, cold_numbers

    def sequence_pattern_analysis(self, sequence_length=3):
        """Analyze patterns of consecutive numbers"""
        sequences = Counter()
        for draw in self.draws:
            numbers = sorted(draw[1])
            for i in range(len(numbers) - sequence_length + 1):
                sequence = tuple(numbers[i:i+sequence_length])
                sequences.update([sequence])
        return sequences.most_common()

    def cluster_analysis(self, n_clusters=3):
        """Cluster numbers based on their frequency"""
        try:
            frequency = self.count_frequency()
            numbers = list(frequency.keys())
            frequencies = list(frequency.values())
            
            if len(numbers) < n_clusters:
                print(f"WARNING: Not enough unique numbers ({len(numbers)}) for {n_clusters} clusters")
                n_clusters = min(len(numbers), 2)
            
            X = np.array(frequencies).reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
            
            clusters = {i: [] for i in range(n_clusters)}
            for number, label in zip(numbers, kmeans.labels_):
                clusters[label].append(number)
                
            print(f"DEBUG: Created {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            print(f"ERROR in cluster_analysis: {e}")
            traceback.print_exc()
            return {0: list(range(1, 81))}  # Fallback to single cluster

    def get_analysis_results(self):
        """Get all analysis results in one call"""
        try:
            results = {
                'frequency': self.count_frequency(),
                'top_numbers': self.get_top_numbers(),
                'common_pairs': self.find_common_pairs(),
                'consecutive': self.find_consecutive_numbers(),
                'ranges': self.number_range_analysis(),
                'hot_cold': self.hot_and_cold_numbers(),
                'sequences': self.sequence_pattern_analysis(),
                'clusters': self.cluster_analysis()
            }
            print(f"DEBUG: Generated complete analysis with {len(results)} components")
            return results
        except Exception as e:
            print(f"ERROR in get_analysis_results: {e}")
            traceback.print_exc()
            return None

    def save_to_excel(self, filename=None):
        """Save analysis results to Excel file using config paths"""
        if filename is None:
            filename = PATHS['ANALYSIS']
        
        try:
            # Ensure directories exist
            ensure_directories()
            
            # Make sure we have a directory path
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Get all analysis results
            frequency = self.count_frequency()
            common_pairs = self.find_common_pairs()
            consecutive_numbers = self.find_consecutive_numbers()
            range_analysis = self.number_range_analysis()
            hot_numbers, cold_numbers = self.hot_and_cold_numbers()
            sequence_patterns = self.sequence_pattern_analysis()
            clusters = self.cluster_analysis()

            # Create DataFrames
            frequency_df = pd.DataFrame(frequency.items(), columns=["Number", "Frequency"])
            
            common_pairs_df = pd.DataFrame(common_pairs, columns=["Pair", "Frequency"])
            common_pairs_df["Number 1"] = common_pairs_df["Pair"].apply(lambda x: x[0])
            common_pairs_df["Number 2"] = common_pairs_df["Pair"].apply(lambda x: x[1])
            common_pairs_df = common_pairs_df.drop(columns=["Pair"])

            consecutive_numbers_df = pd.DataFrame(consecutive_numbers, columns=["Pair", "Frequency"])
            consecutive_numbers_df["Number 1"] = consecutive_numbers_df["Pair"].apply(lambda x: x[0])
            consecutive_numbers_df["Number 2"] = consecutive_numbers_df["Pair"].apply(lambda x: x[1])
            consecutive_numbers_df = consecutive_numbers_df.drop(columns=["Pair"])

            range_analysis_df = pd.DataFrame(range_analysis.items(), columns=["Range", "Count"])

            # Create separate DataFrames for hot and cold numbers
            hot_numbers_df = pd.DataFrame(hot_numbers, columns=["Number", "Frequency"])
            cold_numbers_df = pd.DataFrame(cold_numbers, columns=["Number", "Frequency"])

            sequence_patterns_df = pd.DataFrame(sequence_patterns, columns=["Sequence", "Frequency"])

            clusters_df = pd.DataFrame([(k, v) for k, vs in clusters.items() for v in vs], 
                                     columns=["Cluster", "Number"])

            # Save to Excel with all sheets
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                frequency_df.to_excel(writer, sheet_name='Frequency', index=False)
                common_pairs_df.to_excel(writer, sheet_name='Common Pairs', index=False)
                consecutive_numbers_df.to_excel(writer, sheet_name='Consecutive Numbers', index=False)
                range_analysis_df.to_excel(writer, sheet_name='Number Range', index=False)
                hot_numbers_df.to_excel(writer, sheet_name='Hot Numbers', index=False)
                cold_numbers_df.to_excel(writer, sheet_name='Cold Numbers', index=False)
                sequence_patterns_df.to_excel(writer, sheet_name='Sequence Patterns', index=False)
                clusters_df.to_excel(writer, sheet_name='Clusters', index=False)

            print(f"\nAnalysis results saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # Example draws for testing if real data fails to load
    example_draws = [
        ("20:15 26-02-2025", [1, 2, 2, 9, 12, 14, 17, 25, 26, 30, 38, 44, 54, 57, 58, 61, 65, 71, 72, 76, 79]),
        ("20:10 26-02-2025", [4, 5, 7, 7, 9, 18, 24, 27, 29, 34, 40, 45, 48, 52, 55, 57, 70, 71, 72, 74, 77]),
    ]
    
    try:
        # Try to load real data from the CSV file
        real_draws = []
        csv_file = PATHS['HISTORICAL_DATA']
        
        if os.path.exists(csv_file):
            print(f"\nLoading historical draws from: {csv_file}")
            
            import csv
            with open(csv_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader, None)  # Skip header if exists
                
                for row in csv_reader:
                    if len(row) >= 21:  # Date/time + 20 numbers
                        draw_date = row[0]
                        try:
                            # Convert all number strings to integers
                            numbers = [int(num.strip()) for num in row[1:21] if num.strip()]
                            real_draws.append((draw_date, numbers))
                        except ValueError as e:
                            print(f"Skipping row with invalid numbers: {e}")
            
            if real_draws:
                print(f"Successfully loaded {len(real_draws)} historical draws")
                draws_to_analyze = real_draws
            else:
                print(f"No valid draws found in {csv_file}, using example draws instead")
                draws_to_analyze = example_draws
        else:
            print(f"Historical data file not found: {csv_file}")
            print("Using example draws instead")
            draws_to_analyze = example_draws
            
        # Initialize analysis with loaded data
        analysis = DataAnalysis(draws_to_analyze)
        
        # Run analyses
        frequency = analysis.count_frequency()
        print(f"Number of unique numbers: {len(frequency)}")
        
        top_numbers = analysis.get_top_numbers(20)
        print(f"Top 20 numbers: {', '.join(map(str, top_numbers))}")
        
        # Save results
        analysis.save_to_excel(PATHS['ANALYSIS'])
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
        traceback.print_exc()