from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories, print_system_info

class DataAnalysis:
    def print_system_info():
        """Print system information"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}")

    def safe_parse_date(self, draw_info):
        """
        Parse draw dates in various formats
        
        Args:
            draw_info (tuple): Tuple containing draw date and numbers
            
        Returns:
            datetime: Parsed datetime object
        """
        try:
            draw_date = draw_info[0].strip()
            # First normalize the date string - combine multiple spaces
            draw_date = ' '.join(draw_date.split())
            
            try:
                # Try standard format "HH:MM DD-MM-YYYY"
                parts = draw_date.split(' ', 1)
                if len(parts) == 2:
                    time_part, date_part = parts
                    # Parse date in correct order: day, month, year
                    dt = datetime.strptime(f"{date_part} {time_part}", "%d-%m-%Y %H:%M")
                    return dt
            except ValueError:
                try:
                    # Try direct parsing with the expected format
                    dt = datetime.strptime(draw_date, "%H:%M %d-%m-%Y")
                    return dt
                except ValueError:
                    # Try alternative date formats
                    formats = [
                        "%d-%m-%Y %H:%M",  # Day-Month-Year Hour:Minute
                        "%m-%d-%Y %H:%M",  # Month-Day-Year Hour:Minute
                        "%Y-%m-%d %H:%M"   # Year-Month-Day Hour:Minute
                    ]
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(draw_date, fmt)
                            return dt
                        except ValueError:
                            continue
            
            # If we get here, no formats worked
            print(f"WARNING: Could not parse date '{draw_date}', using minimum date")
            return datetime.min
        except Exception as e:
            print(f"ERROR parsing date '{draw_info[0]}': {e}")
            return datetime.min

    def __init__(self, draws=None, debug=False):
        """Initialize DataAnalysis with proper draw validation"""
        self.debug = debug
        if draws is None:
            self.draws = self.load_historical_data(PATHS['HISTORICAL_DATA'])
        else:
            self.draws = []
            
        if self.debug:
            print(f"\nDEBUG: Initializing DataAnalysis with {len(draws if draws else [])} draws")
            print(f"DEBUG: First draw format: {draws[0] if draws else 'None'}")
        
        # Process draws if provided directly
        if draws:
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
        
        # Print sample of dates before sorting
        if self.debug and self.draws:
            print("\nDEBUG: Sample of dates before sorting:")
            for i in range(min(3, len(self.draws))):
                print(f"  {self.draws[i][0]}")
        
        # Sort draws by date in chronological order with robust parsing
        try:
            self.draws.sort(key=self.safe_parse_date)
            
            if self.debug:
                print(f"\nDEBUG: Sorted {len(self.draws)} draws by date")
                print(f"DEBUG: First draw (oldest): {self.draws[0][0] if self.draws else 'None'}")
                print(f"DEBUG: Last draw (newest): {self.draws[-1][0] if self.draws else 'None'}")
                
                # Print the last few draws to verify sorting
                if len(self.draws) > 5:
                    print("\nDEBUG: Last 5 draws (most recent):")
                    for i in range(1, 6):
                        if i <= len(self.draws):
                            print(f"  {self.draws[-i][0]}")
        except Exception as e:
            print(f"ERROR during draw sorting: {e}")
            traceback.print_exc()

        if not self.draws:
            raise ValueError("No valid draws were processed!")

    def load_historical_data(self, data_path):
        """Load historical data from CSV file"""
        draws = []
        try:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as file:
                    import csv
                    csv_reader = csv.reader(file)
                    header = next(csv_reader, None)  # Skip header if exists
                    
                    for row in csv_reader:
                        if len(row) >= 21:  # Date/time + 20 numbers
                            draw_date = row[0]
                            try:
                                numbers = [int(num.strip()) for num in row[1:21] if num.strip()]
                                draws.append((draw_date, numbers))
                            except ValueError as e:
                                print(f"Skipping row with invalid numbers: {e}")
            else:
                print(f"Historical data file not found: {data_path}")
        except Exception as e:
            print(f"Error loading historical data: {e}")
            traceback.print_exc()
        return draws

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
    
    def hot_and_cold_numbers(self, top_n=10, window_size=24):
        """Enhanced hot/cold analysis with trending detection"""
        # Overall hot/cold
        frequency = self.count_frequency()
        sorted_numbers = frequency.most_common()
        
        # Recent trends (using window)
        recent_draws = self.draws[-window_size:] if len(self.draws) > window_size else self.draws
        recent_numbers = [number for _, numbers in recent_draws for number in numbers]
        recent_frequency = Counter(recent_numbers)
        
        # Calculate trend (increasing/decreasing frequency)
        trends = {}
        for num in range(1, 81):
            overall_freq = frequency[num] / max(len(self.draws), 1) if self.draws else 0
            recent_freq = recent_frequency[num] / max(len(recent_draws), 1) if recent_draws else 0
            trend = recent_freq - overall_freq
            trends[num] = {
                'overall_freq': round(overall_freq, 4),
                'recent_freq': round(recent_freq, 4),
                'trend': round(trend, 4),
                'number': num
            }
        
        # Get trending up and down numbers
        trending_up = sorted([(k, v) for k, v in trends.items()], key=lambda x: x[1]['trend'], reverse=True)[:top_n]
        trending_down = sorted([(k, v) for k, v in trends.items()], key=lambda x: x[1]['trend'])[:top_n]
        
        if self.debug:
            print(f"DEBUG: Analyzed hot/cold numbers with window size {window_size}")
            print(f"DEBUG: Found {len(trending_up)} trending up and {len(trending_down)} trending down numbers")
        
        return {
            'hot_numbers': sorted_numbers[:top_n],
            'cold_numbers': sorted_numbers[-top_n:],
            'trending_up': trending_up,
            'trending_down': trending_down
        }

    def sequence_pattern_analysis(self, sequence_length=3):
        """Enhanced sequence analysis with detailed time patterns"""
        sequences = Counter()
        time_patterns = {
            'hourly': defaultdict(Counter),
            'daily': defaultdict(Counter),
            'five_minute': defaultdict(Counter)
        }
        
        for i in range(len(self.draws) - sequence_length + 1):
            window = self.draws[i:i+sequence_length]
            numbers_sequence = tuple(sorted(window[-1][1]))
            sequences.update([numbers_sequence])
            
            draw_time = window[-1][0]
            try:
                time_parts = draw_time.split()
                hour_min = time_parts[0].split(':')
                hour = int(hour_min[0])
                minute = int(hour_min[1])
                
                date_parts = time_parts[1].split('-')
                draw_datetime = datetime(
                    year=int(date_parts[2]),
                    month=int(date_parts[1]),
                    day=int(date_parts[0]),
                    hour=hour,
                    minute=minute
                )
                
                time_patterns['hourly'][hour].update([numbers_sequence])
                time_patterns['daily'][draw_datetime.weekday()].update([numbers_sequence])
                
                five_min_interval = 5 * (minute // 5)
                time_key = f"{hour:02d}:{five_min_interval:02d}"
                time_patterns['five_minute'][time_key].update([numbers_sequence])
                
            except (ValueError, IndexError) as e:
                if self.debug:
                    print(f"Error parsing time {draw_time}: {e}")
                continue
        
        time_analysis = {
            'hourly_favorites': {
                hour: {
                    'most_common': seqs.most_common(5),
                    'total_draws': sum(seqs.values())
                }
                for hour, seqs in time_patterns['hourly'].items()
            },
            'daily_favorites': {
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]: {
                    'most_common': seqs.most_common(5),
                    'total_draws': sum(seqs.values())
                }
                for day, seqs in time_patterns['daily'].items()
            },
            'time_slot_favorites': {
                time_slot: {
                    'most_common': seqs.most_common(5),
                    'total_draws': sum(seqs.values())
                }
                for time_slot, seqs in time_patterns['five_minute'].items()
            }
        }
        
        most_active_hour = max(time_patterns['hourly'].items(), 
                              key=lambda x: sum(x[1].values()))[0] if time_patterns['hourly'] else None
        
        most_active_day = None
        if time_patterns['daily']:
            day_index = max(time_patterns['daily'].items(),
                          key=lambda x: sum(x[1].values()))[0]
            most_active_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                              'Friday', 'Saturday', 'Sunday'][day_index]
        
        most_active_time_slot = max(time_patterns['five_minute'].items(),
                                  key=lambda x: sum(x[1].values()))[0] if time_patterns['five_minute'] else None
        
        return {
            'overall_sequences': sequences.most_common(),
            'time_analysis': time_analysis,
            'statistics': {
                'total_sequences': sum(sequences.values()),
                'unique_sequences': len(sequences),
                'most_active_hour': most_active_hour,
                'most_active_day': most_active_day,
                'most_active_time_slot': most_active_time_slot
            }
        }

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
            return {0: list(range(1, 81))}

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
           filename = PATHS['ANALYSIS_RESULTS'] 
        
        try:
            ensure_directories()
            
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
    
            # Get all analysis results
            frequency = self.count_frequency()
            common_pairs = self.find_common_pairs()
            consecutive_numbers = self.find_consecutive_numbers()
            range_analysis = self.number_range_analysis()
            
            # Update to use new hot_and_cold_numbers return format
            hot_cold_data = self.hot_and_cold_numbers()
            hot_numbers = hot_cold_data['hot_numbers']
            cold_numbers = hot_cold_data['cold_numbers']
            trending_up = hot_cold_data['trending_up']
            trending_down = hot_cold_data['trending_down']
            
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
    
            # Create DataFrames for hot/cold analysis
            hot_numbers_df = pd.DataFrame(hot_numbers, columns=["Number", "Frequency"])
            cold_numbers_df = pd.DataFrame(cold_numbers, columns=["Number", "Frequency"])
            
            # Add new DataFrames for trending analysis
            trending_up_df = pd.DataFrame([
                {"Number": num, "Overall_Freq": data['overall_freq'], 
                 "Recent_Freq": data['recent_freq'], "Trend": data['trend']} 
                for num, data in trending_up
            ])
            
            trending_down_df = pd.DataFrame([
                {"Number": num, "Overall_Freq": data['overall_freq'], 
                 "Recent_Freq": data['recent_freq'], "Trend": data['trend']} 
                for num, data in trending_down
            ])
    
            # Create DataFrame for sequence patterns
            if isinstance(sequence_patterns, dict) and 'overall_sequences' in sequence_patterns:
                sequence_patterns_df = pd.DataFrame(sequence_patterns['overall_sequences'], 
                                                 columns=["Sequence", "Frequency"])
                
                if 'time_analysis' in sequence_patterns and 'hourly_favorites' in sequence_patterns['time_analysis']:
                    time_analysis_df = pd.DataFrame([
                        {
                            'Hour': hour,
                            'Most_Common_Sequence': str(data['most_common'][0][0]) if data['most_common'] else 'None',
                            'Frequency': data['most_common'][0][1] if data['most_common'] else 0,
                            'Total_Draws': data['total_draws']
                        }
                        for hour, data in sequence_patterns['time_analysis']['hourly_favorites'].items()
                    ])
            else:
                sequence_patterns_df = pd.DataFrame(sequence_patterns, columns=["Sequence", "Frequency"])
                time_analysis_df = pd.DataFrame(columns=['Hour', 'Most_Common_Sequence', 'Frequency', 'Total_Draws'])
    
            clusters_df = pd.DataFrame([(k, v) for k, vs in clusters.items() for v in vs], 
                                     columns=["Cluster", "Number"])
    
            # Create DataFrame for gap analysis
            gap_stats = self.analyze_gaps()
            gap_analysis_df = pd.DataFrame([
                {
                    'Number': num,
                    'Average_Gap': stats['avg_gap'],
                    'Max_Gap': stats['max_gap'],
                    'Min_Gap': stats['min_gap'],
                    'Current_Gap': stats['current_gap']
                }
                for num, stats in gap_stats.items()
            ])
    
            # Get combinations analysis
            combinations_analysis = self.analyze_combinations(group_size=3, top_n=10)
            
            # Create DataFrames for combinations analysis
            most_common_combinations_df = pd.DataFrame([
                {
                    'Combination': str(list(item['combination'])).replace('[', '').replace(']', ''),
                    'Frequency': item['frequency'],
                    'Percentage': round(item['percentage'], 2),
                    'Average_Gap': round(item['average_gap'], 2)
                }
                for item in combinations_analysis['most_common']
            ])
            
            least_common_combinations_df = pd.DataFrame([
                {
                    'Combination': str(list(item['combination'])).replace('[', '').replace(']', ''),
                    'Frequency': item['frequency'],
                    'Percentage': round(item['percentage'], 2),
                    'Average_Gap': round(item['average_gap'], 2)
                }
                for item in combinations_analysis['least_common']
            ])
            
            # Create statistics dataframe for combinations
            combinations_stats_df = pd.DataFrame([{
                'Total_Combinations': combinations_analysis['statistics']['total_combinations'],
                'Total_Occurrences': combinations_analysis['statistics']['total_occurrences'],
                'Average_Frequency': combinations_analysis['statistics']['avg_frequency'],
                'Group_Size': combinations_analysis['statistics']['group_size']
            }])
    
            # Add skip patterns analysis
            skip_analysis = self.analyze_skip_patterns()
    
            # Create DataFrames for skip patterns
            skip_patterns_df = pd.DataFrame([
                {
                    'Draw Date': pattern['draw_date'],
                    'Skipped Count': pattern['skipped_count'],
                    'New Count': pattern['new_count'],
                    'Repeat Count': pattern['repeat_count'],
                    'Skipped Numbers': str(pattern['skipped_numbers']).replace('[', '').replace(']', ''),
                    'New Numbers': str(pattern['new_numbers']).replace('[', '').replace(']', ''),
                    'Repeated Numbers': str(pattern['repeated_numbers']).replace('[', '').replace(']', '')
                }
                for pattern in skip_analysis['patterns']
            ])
    
            # Create stats DataFrame for skip patterns
            skip_patterns_stats_df = pd.DataFrame([{
                'Average Skipped': skip_analysis['statistics']['avg_skipped'],
                'Average New': skip_analysis['statistics']['avg_new'],
                'Most Volatile Date': skip_analysis['statistics']['most_volatile_dates'][0]['draw_date'] if skip_analysis['statistics']['most_volatile_dates'] else 'N/A',
                'Most Stable Date': skip_analysis['statistics']['most_stable_dates'][0]['draw_date'] if skip_analysis['statistics']['most_stable_dates'] else 'N/A'
            }])
    
            # Get statistical significance results
            significance_results = self.analyze_statistical_significance()
            if significance_results and 'frequency_tests' in significance_results:
                significance_df = pd.DataFrame(significance_results['frequency_tests'])
                pattern_tests_df = pd.DataFrame(significance_results['pattern_tests']) if 'pattern_tests' in significance_results and significance_results['pattern_tests'] else pd.DataFrame()
                time_tests_df = pd.DataFrame(significance_results['time_tests']) if 'time_tests' in significance_results and significance_results['time_tests'] else pd.DataFrame()
            else:
                significance_df = pd.DataFrame()
                pattern_tests_df = pd.DataFrame()
                time_tests_df = pd.DataFrame()
                
            # Get cross-validation results
            validation_results = self.validate_patterns()
            if validation_results and 'fold_results' in validation_results:
                validation_folds_df = pd.DataFrame(validation_results['fold_results'])
                validation_summary_df = pd.DataFrame([validation_results['validation_summary']])
                
                # Create DataFrame for consistent patterns
                consistent_hot_numbers_df = pd.DataFrame(
                    validation_results['pattern_stability']['consistent_hot_numbers']
                    if 'pattern_stability' in validation_results else []
                )
                consistent_pairs_df = pd.DataFrame(
                    validation_results['pattern_stability']['consistent_pairs']
                    if 'pattern_stability' in validation_results else []
                )
                consistent_trends_df = pd.DataFrame(
                    validation_results['pattern_stability']['consistent_trends']
                    if 'pattern_stability' in validation_results else []
                )
            else:
                validation_folds_df = pd.DataFrame()
                validation_summary_df = pd.DataFrame()
                consistent_hot_numbers_df = pd.DataFrame()
                consistent_pairs_df = pd.DataFrame()
                consistent_trends_df = pd.DataFrame()
            
            # Add new sheet for focused predictions
            focused_predictions = self.get_focused_prediction()
            if focused_predictions:
                focused_df = pd.DataFrame({
                    'Predicted Numbers': [str(focused_predictions['numbers'])],
                    'Confidence Scores': [str(focused_predictions['confidence'])],
                    'Timestamp': [focused_predictions['timestamp']]
                })
            else:
                focused_df = pd.DataFrame(columns=['Predicted Numbers', 'Confidence Scores', 'Timestamp'])
    
            # Save to Excel with all sheets
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                frequency_df.to_excel(writer, sheet_name='Frequency', index=False)
                common_pairs_df.to_excel(writer, sheet_name='Common Pairs', index=False)
                consecutive_numbers_df.to_excel(writer, sheet_name='Consecutive Numbers', index=False)
                range_analysis_df.to_excel(writer, sheet_name='Number Range', index=False)
                hot_numbers_df.to_excel(writer, sheet_name='Hot Numbers', index=False)
                cold_numbers_df.to_excel(writer, sheet_name='Cold Numbers', index=False)
                trending_up_df.to_excel(writer, sheet_name='Trending Up', index=False)
                trending_down_df.to_excel(writer, sheet_name='Trending Down', index=False)
                sequence_patterns_df.to_excel(writer, sheet_name='Sequence Patterns', index=False)
                time_analysis_df.to_excel(writer, sheet_name='Time Analysis', index=False)
                clusters_df.to_excel(writer, sheet_name='Clusters', index=False)
                gap_analysis_df.to_excel(writer, sheet_name='Gap Analysis', index=False)
                most_common_combinations_df.to_excel(writer, sheet_name='Most Common Combinations', index=False)
                least_common_combinations_df.to_excel(writer, sheet_name='Least Common Combinations', index=False)
                combinations_stats_df.to_excel(writer, sheet_name='Combinations Statistics', index=False)
                skip_patterns_df.to_excel(writer, sheet_name='Skip Patterns', index=False)
                skip_patterns_stats_df.to_excel(writer, sheet_name='Skip Pattern Stats', index=False)
                significance_df.to_excel(writer, sheet_name='Significance Tests', index=False)
                pattern_tests_df.to_excel(writer, sheet_name='Pattern Tests', index=False)
                time_tests_df.to_excel(writer, sheet_name='Time Tests', index=False)
                validation_folds_df.to_excel(writer, sheet_name='Validation Folds', index=False)
                validation_summary_df.to_excel(writer, sheet_name='Validation Summary', index=False)
                consistent_hot_numbers_df.to_excel(writer, sheet_name='Consistent Hot Numbers', index=False)
                consistent_pairs_df.to_excel(writer, sheet_name='Consistent Pairs', index=False)
                consistent_trends_df.to_excel(writer, sheet_name='Consistent Trends', index=False)
                focused_df.to_excel(writer, sheet_name='Focused_Predictions', index=False)
    
            print(f"\nAnalysis results saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving analysis results: {e}")
            traceback.print_exc()
            return False

    def analyze_gaps(self, num_range=(1, 80)):
        """
        Analyze gaps between appearances for each number
        
        Args:
            num_range (tuple): Range of numbers to analyze (min, max)
            
        Returns:
            dict: Dictionary containing gap analysis for each number with structure:
                {
                    number: {
                        'avg_gap': average gap between appearances,
                        'max_gap': longest gap between appearances,
                        'min_gap': shortest gap between appearances,
                        'current_gap': draws since last appearance,
                        'gap_history': list of all gaps
                    }
                }
        """
        # Initialize tracking dictionaries
        gaps = defaultdict(list)
        last_seen = defaultdict(int)
        current_draw_index = 0
        
        # Track gaps for each number
        for draw_index, (draw_date, numbers) in enumerate(self.draws):
            current_draw_index = draw_index
            
            # Check each possible number
            for num in range(num_range[0], num_range[1] + 1):
                if num in numbers:
                    # If we've seen this number before, calculate gap
                    if last_seen[num] > 0:
                        gap = draw_index - last_seen[num]
                        gaps[num].append(gap)
                    last_seen[num] = draw_index
        
        # Calculate statistics for each number
        gap_stats = {}
        for num in range(num_range[0], num_range[1] + 1):
            if gaps[num]:  # Only include numbers that have appeared at least twice
                gap_stats[num] = {
                    'avg_gap': round(np.mean(gaps[num]), 2),
                    'max_gap': max(gaps[num]),
                    'min_gap': min(gaps[num]),
                    'current_gap': current_draw_index - last_seen[num] if last_seen[num] > 0 else current_draw_index + 1,
                    'gap_history': gaps[num]
                }
            else:  # Handle numbers that appeared once or never
                gap_stats[num] = {
                    'avg_gap': current_draw_index + 1,
                    'max_gap': current_draw_index + 1,
                    'min_gap': current_draw_index + 1,
                    'current_gap': current_draw_index - last_seen[num] if last_seen[num] > 0 else current_draw_index + 1,
                    'gap_history': []
                }
        
        if self.debug:
            print(f"DEBUG: Analyzed gaps for {len(gap_stats)} numbers")
            
        return gap_stats

    def analyze_combinations(self, group_size=3, top_n=10):
        """
        Analyze frequently occurring number combinations
        
        Args:
            group_size (int): Size of number combinations to analyze (2-5)
            top_n (int): Number of top/bottom combinations to return
            
        Returns:
            dict: Dictionary containing combination analysis
        """
        try:
            # Validate inputs
            if not self.draws:
                raise ValueError("No draws available for analysis")
                
            group_size = max(2, min(5, group_size))  # Ensure group_size is between 2 and 5
            top_n = max(1, min(50, top_n))  # Ensure top_n is between 1 and 50
            
            if self.debug:
                print(f"DEBUG: Analyzing combinations of size {group_size}")
            
            # Track combinations
            combinations_count = Counter()
            total_draws = len(self.draws)
            
            # Process each draw
            for _, numbers in self.draws:
                if len(numbers) >= group_size:  # Ensure we have enough numbers
                    # Get all possible combinations of the specified size
                    combos = combinations(sorted(numbers), group_size)
                    combinations_count.update(combos)
            
            if not combinations_count:
                raise ValueError(f"No valid combinations of size {group_size} found")
                
            # Calculate statistics
            most_common = combinations_count.most_common(top_n)
            least_common = sorted(combinations_count.items(), key=lambda x: x[1])[:top_n]
            
            # Calculate averages and percentages
            total_combinations = len(combinations_count)
            total_occurrences = sum(combinations_count.values())
            avg_frequency = total_occurrences / total_combinations if total_combinations > 0 else 0
            
            if self.debug:
                print(f"DEBUG: Found {total_combinations} unique combinations")
                print(f"DEBUG: Average frequency: {avg_frequency:.2f}")
            
            return {
                'most_common': [
                    {
                        'combination': combo,
                        'frequency': freq,
                        'percentage': round((freq / total_draws) * 100, 2),
                        'average_gap': round(total_draws / freq, 2) if freq > 0 else total_draws
                    }
                    for combo, freq in most_common
                ],
                'least_common': [
                    {
                        'combination': combo,
                        'frequency': freq,
                        'percentage': round((freq / total_draws) * 100, 2),
                        'average_gap': round(total_draws / freq, 2) if freq > 0 else total_draws
                    }
                    for combo, freq in least_common
                ],
                'statistics': {
                    'total_combinations': total_combinations,
                    'total_occurrences': total_occurrences,
                    'avg_frequency': round(avg_frequency, 2),
                    'group_size': group_size
                }
            }
            
        except Exception as e:
            print(f"ERROR in analyze_combinations: {e}")
            traceback.print_exc()
            return {
                'most_common': [],
                'least_common': [],
                'statistics': {
                    'total_combinations': 0,
                    'total_occurrences': 0,
                    'avg_frequency': 0,
                    'group_size': group_size
                }
            }

    def analyze_skip_patterns(self, window_size=6):
        """
        Analyze patterns in number skips between consecutive draws
        
        Args:
            window_size (int): Number of recent draws to analyze in detail
            
        Returns:
            dict: Dictionary containing skip pattern analysis
        """
        try:
            if len(self.draws) < 2:
                raise ValueError("Need at least 2 draws to analyze skip patterns")
                
            skip_patterns = []
            total_skips = 0
            total_new_numbers = 0
            
            # Analyze consecutive draws
            for i in range(len(self.draws) - 1):
                current_numbers = set(self.draws[i][1])
                next_numbers = set(self.draws[i+1][1])
                
                # Calculate skips and new numbers
                skipped = current_numbers - next_numbers
                new_numbers = next_numbers - current_numbers
                repeated = current_numbers.intersection(next_numbers)
                
                pattern = {
                    'draw_date': self.draws[i+1][0],
                    'skipped_count': len(skipped),
                    'new_count': len(new_numbers),
                    'repeat_count': len(repeated),
                    'skipped_numbers': sorted(skipped),
                    'new_numbers': sorted(new_numbers),
                    'repeated_numbers': sorted(repeated)
                }
                
                skip_patterns.append(pattern)
                total_skips += len(skipped)
                total_new_numbers += len(new_numbers)
            
            # Calculate recent patterns
            recent_patterns = skip_patterns[-window_size:] if len(skip_patterns) > window_size else skip_patterns
            
            # Calculate statistics
            total_draws = len(skip_patterns)
            stats = {
                'avg_skipped': round(total_skips / total_draws, 2) if total_draws > 0 else 0,
                'avg_new': round(total_new_numbers / total_draws, 2) if total_draws > 0 else 0,
                'most_volatile_dates': sorted(skip_patterns, 
                                           key=lambda x: x['skipped_count'] + x['new_count'], 
                                           reverse=True)[:5],
                'most_stable_dates': sorted(skip_patterns, 
                                         key=lambda x: x['repeat_count'], 
                                         reverse=True)[:5]
            }
            
            if self.debug:
                print(f"DEBUG: Analyzed {len(skip_patterns)} skip patterns")
                print(f"DEBUG: Average numbers skipped: {stats['avg_skipped']}")
                print(f"DEBUG: Average new numbers: {stats['avg_new']}")
            
            return {
                'patterns': skip_patterns,
                'statistics': stats,
                'recent_patterns': recent_patterns
            }
            
        except Exception as e:
            print(f"ERROR in analyze_skip_patterns: {e}")
            traceback.print_exc()
            return {
                'patterns': [],
                'statistics': {
                    'avg_skipped': 0,
                    'avg_new': 0,
                    'most_volatile_dates': [],
                    'most_stable_dates': []
                },
                'recent_patterns': []
            }

    def analyze_statistical_significance(self):
        """
        Analyze statistical significance of patterns
        
        Returns:
            dict: Dictionary containing statistical test results for frequencies and patterns
        """
        try:
            from scipy import stats
            
            results = {
                'frequency_tests': [],
                'pattern_tests': [],
                'time_tests': []
            }
            
            # Expected frequency if uniform distribution
            expected_per_draw = 20/80  # Each number has 20/80 chance per draw
            total_draws = len(self.draws)
            expected_frequency = total_draws * expected_per_draw
            
            # Test frequency significance
            frequency = self.count_frequency()
            for number, observed_freq in frequency.items():
                chi_square = ((observed_freq - expected_frequency) ** 2) / expected_frequency
                p_value = 1 - stats.chi2.cdf(chi_square, df=1)
                
                results['frequency_tests'].append({
                    'number': number,
                    'observed': observed_freq,
                    'expected': expected_frequency,
                    'chi_square': round(chi_square, 4),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05
                })
                
            # Test for patterns in time series (runs test)
            try:
                # For each number, create a binary time series (1=appeared, 0=didn't appear)
                for number in range(1, 81):
                    binary_series = [1 if number in draw[1] else 0 for draw in self.draws]
                    
                    # Need at least a few occurrences for meaningful analysis
                    if sum(binary_series) > 5 and sum(binary_series) < len(binary_series) - 5:
                        runs, n1, n0 = stats.runstest_1samp(binary_series)
                        # Convert runs to p-value
                        p_value = 2 * (1 - stats.norm.cdf(abs(runs)))
                        
                        results['pattern_tests'].append({
                            'number': number,
                            'runs_test': round(runs, 4),
                            'p_value': round(p_value, 4),
                            'significant': p_value < 0.05,
                            'interpretation': "Non-random" if p_value < 0.05 else "Random"
                        })
            except AttributeError:
                # If runstest is not available in the scipy version
                if self.debug:
                    print("DEBUG: Runs test not available in this scipy version")
                pass
                
            # Analyze hot/cold patterns over time
            try:
                hot_cold = self.hot_and_cold_numbers()
                trending_up = [num for num, data in hot_cold['trending_up']]
                trending_down = [num for num, data in hot_cold['trending_down']]
                
                # Test if trending patterns are statistically significant
                # Using binomial test to see if trend deviates from random chance
                for num in range(1, 81):
                    if num in trending_up or num in trending_down:
                        # Get recent vs overall frequency
                        window_size = min(50, len(self.draws))
                        recent_draws = self.draws[-window_size:]
                        
                        # Count occurrences
                        recent_count = sum(1 for _, numbers in recent_draws if num in numbers)
                        total_count = frequency[num]
                        
                        # Expected probability based on overall data
                        expected_prob = total_count / len(self.draws)
                        
                        # Binomial test for recent data
                        p_value = stats.binom_test(recent_count, n=window_size, p=expected_prob)
                        
                        results['time_tests'].append({
                            'number': num,
                            'trend': "Up" if num in trending_up else "Down",
                            'recent_frequency': recent_count / window_size,
                            'overall_frequency': expected_prob,
                            'p_value': round(p_value, 4),
                            'significant': p_value < 0.05
                        })
            except Exception as e:
                if self.debug:
                    print(f"DEBUG: Error in time trend analysis: {e}")
                    
            if self.debug:
                significant_freq = sum(1 for t in results['frequency_tests'] if t['significant'])
                significant_patterns = sum(1 for t in results['pattern_tests'] if t['significant'])
                print(f"DEBUG: Found {significant_freq} statistically significant frequencies")
                print(f"DEBUG: Found {significant_patterns} statistically significant patterns")
                
            return results
            
        except ImportError:
            print("Warning: scipy not installed. Statistical tests unavailable.")
            return {'error': 'scipy library not installed'}
        except Exception as e:
            print(f"Error in statistical significance analysis: {e}")
            traceback.print_exc()
            return None

    def validate_patterns(self, k_folds=5):
        """
        Cross-validate patterns using k-fold validation
        
        Args:
            k_folds (int): Number of folds to use in cross-validation
            
        Returns:
            dict: Dictionary containing validation results and stability metrics
        """
        try:
            results = {
                'fold_results': [],
                'pattern_stability': {},
                'validation_summary': {}
            }
            
            # Ensure we have enough data for k-fold validation
            total_draws = len(self.draws)
            if total_draws < k_folds * 2:
                k_folds = max(2, total_draws // 2)
                if self.debug:
                    print(f"DEBUG: Not enough draws for {k_folds} folds, reducing to {k_folds}")
            
            # Split data into k folds
            fold_size = total_draws // k_folds
            
            # Track pattern stability across folds
            all_hot_numbers = []
            all_pairs = []
            all_trends_up = []
            
            for fold in range(k_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                # Create training and validation sets
                validation_draws = self.draws[start_idx:end_idx]
                training_draws = self.draws[:start_idx] + self.draws[end_idx:]
                
                # Skip if either set is too small
                if len(training_draws) < 5 or len(validation_draws) < 5:
                    continue
                
                # Analyze each set
                training_analysis = DataAnalysis(training_draws, debug=False)
                validation_analysis = DataAnalysis(validation_draws, debug=False)
                
                # Compare hot numbers
                train_hot = set(num for num, _ in training_analysis.hot_and_cold_numbers()['hot_numbers'][:10])
                val_hot = set(num for num, _ in validation_analysis.hot_and_cold_numbers()['hot_numbers'][:10])
                hot_overlap = len(train_hot.intersection(val_hot))
                all_hot_numbers.append(train_hot)
                
                # Compare common pairs
                train_pairs = set(pair for pair, _ in training_analysis.find_common_pairs(10))
                val_pairs = set(pair for pair, _ in validation_analysis.find_common_pairs(10))
                pair_overlap = len(train_pairs.intersection(val_pairs))
                all_pairs.append(train_pairs)
                
                # Compare trending numbers
                train_trending_up = set(num for num, _ in training_analysis.hot_and_cold_numbers()['trending_up'][:5])
                val_trending_up = set(num for num, _ in validation_analysis.hot_and_cold_numbers()['trending_up'][:5])
                trend_overlap = len(train_trending_up.intersection(val_trending_up))
                all_trends_up.append(train_trending_up)
                
                # Calculate prediction accuracy
                # For each validation draw, check if training hot numbers appear more frequently
                validation_hot_hits = 0
                validation_draws_count = 0
                
                for _, numbers in validation_draws:
                    validation_draws_count += 1
                    # Count hot numbers that appeared
                    hot_hits = sum(1 for num in numbers if num in train_hot)
                    validation_hot_hits += hot_hits
                
                # Average hit rate per draw
                avg_hot_hits = validation_hot_hits / validation_draws_count if validation_draws_count > 0 else 0
                expected_hits = 10 * (20 / 80)  # Expected hits if random
                
                results['fold_results'].append({
                    'fold': fold + 1,
                    'training_size': len(training_draws),
                    'validation_size': len(validation_draws),
                    'hot_numbers_overlap': hot_overlap,
                    'hot_numbers_overlap_percentage': round((hot_overlap / 10) * 100, 2),
                    'common_pairs_overlap': pair_overlap,
                    'common_pairs_overlap_percentage': round((pair_overlap / 10) * 100, 2) if train_pairs else 0,
                    'trend_up_overlap': trend_overlap,
                    'trend_up_overlap_percentage': round((trend_overlap / 5) * 100, 2) if train_trending_up else 0,
                    'hot_number_hit_rate': round(avg_hot_hits, 2),
                    'expected_random_hit_rate': round(expected_hits, 2),
                    'hit_rate_vs_random': round((avg_hot_hits / expected_hits if expected_hits > 0 else 0), 2)
                })
            
            # Calculate consistency across all folds
            # For hot numbers: how many numbers appear in multiple fold analyses
            hot_number_counts = Counter()
            for hot_set in all_hot_numbers:
                hot_number_counts.update(hot_set)
                
            pair_counts = Counter()
            for pair_set in all_pairs:
                pair_counts.update(pair_set)
                
            trend_counts = Counter()
            for trend_set in all_trends_up:
                trend_counts.update(trend_set)
                
            # Calculate stability metrics
            results['pattern_stability'] = {
                'consistent_hot_numbers': [
                    {'number': num, 'fold_occurrences': count, 'percentage': round((count / k_folds) * 100, 2)}
                    for num, count in hot_number_counts.most_common()
                    if count > 1  # Appears in more than one fold
                ],
                'consistent_pairs': [
                    {'pair': pair, 'fold_occurrences': count, 'percentage': round((count / k_folds) * 100, 2)}
                    for pair, count in pair_counts.most_common(5)
                    if count > 1
                ],
                'consistent_trends': [
                    {'number': num, 'fold_occurrences': count, 'percentage': round((count / k_folds) * 100, 2)}
                    for num, count in trend_counts.most_common(5)
                    if count > 1
                ]
            }
            
            # Calculate overall stability scores
            avg_hot_overlap = sum(r['hot_numbers_overlap_percentage'] for r in results['fold_results']) / len(results['fold_results']) if results['fold_results'] else 0
            avg_pair_overlap = sum(r['common_pairs_overlap_percentage'] for r in results['fold_results']) / len(results['fold_results']) if results['fold_results'] else 0
            avg_trend_overlap = sum(r['trend_up_overlap_percentage'] for r in results['fold_results']) / len(results['fold_results']) if results['fold_results'] else 0
            avg_hit_rate = sum(r['hot_number_hit_rate'] for r in results['fold_results']) / len(results['fold_results']) if results['fold_results'] else 0
            avg_random_rate = sum(r['expected_random_hit_rate'] for r in results['fold_results']) / len(results['fold_results']) if results['fold_results'] else 0
            
            results['validation_summary'] = {
                'average_hot_numbers_stability': round(avg_hot_overlap, 2),
                'average_common_pairs_stability': round(avg_pair_overlap, 2),
                'average_trend_stability': round(avg_trend_overlap, 2),
                'average_hit_rate': round(avg_hit_rate, 2),
                'average_expected_random_rate': round(avg_random_rate, 2),
                'hit_rate_advantage': round(avg_hit_rate - avg_random_rate, 2),
                'number_of_folds': k_folds,
                'total_draws_analyzed': total_draws,
                'most_consistent_number': results['pattern_stability']['consistent_hot_numbers'][0]['number'] 
                    if results['pattern_stability']['consistent_hot_numbers'] else None,
                'predictive_power': "Above Random" if avg_hit_rate > avg_random_rate else 
                                   "Below Random" if avg_hit_rate < avg_random_rate else "Random"
            }
            
            if self.debug:
                print(f"DEBUG: Completed {k_folds}-fold cross-validation")
                print(f"DEBUG: Hot number stability: {avg_hot_overlap:.2f}%")
                print(f"DEBUG: Predictive power: {results['validation_summary']['predictive_power']}")
                
            return results
            
        except Exception as e:
            print(f"Error in pattern validation: {e}")
            traceback.print_exc()
            return None

    def get_focused_prediction(self, top_n=15):
        """
        Get focused prediction of top N numbers
        
        Args:
            top_n (int): Number of predictions to return
            
        Returns:
            dict: Dictionary containing predicted numbers, confidence scores, and timestamp
        """
        try:
            # Combine multiple analysis methods for scoring
            scores = {}
            
            # 1. Frequency Analysis (30%)
            frequency = self.count_frequency()
            max_freq = max(frequency.values()) if frequency else 1
            
            # 2. Recent Performance (30%)
            hot_cold = self.hot_and_cold_numbers(top_n=80)  # Get all numbers
            
            # 3. Pattern Strength (40%)
            gaps = self.analyze_gaps()
            
            # Calculate combined scores
            for num in range(1, 81):
                # Frequency score
                freq_score = (frequency.get(num, 0) / max_freq) * 0.30
                
                # Recent performance score
                recent_score = 0
                for hot_num, _ in hot_cold['hot_numbers']:
                    if hot_num == num:
                        recent_score = 0.30
                        break
                
                # Pattern score
                gap_data = gaps.get(num, {})
                if gap_data:
                    current_gap = gap_data.get('current_gap', 0)
                    avg_gap = gap_data.get('avg_gap', 0)
                    pattern_score = 0.40 if current_gap > avg_gap else 0.20
                else:
                    pattern_score = 0
                    
                # Combined score
                scores[num] = freq_score + recent_score + pattern_score
                
            # Get top N numbers with their confidence scores
            sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_numbers = sorted_numbers[:top_n]
            
            return {
                'numbers': [num for num, _ in top_numbers],
                'confidence': [round(score, 3) for _, score in top_numbers],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"Error in focused prediction: {e}")
            traceback.print_exc()
            return None

    def check_prediction_health(self):
        """Monitor prediction system health"""
        health_status = {
            'needs_attention': False,
            'reasons': []
        }
        
        try:
            # 1. Check pattern stability
            validation_results = self.validate_patterns()
            if validation_results:
                stability = validation_results['validation_summary']['average_hot_numbers_stability']
                if stability < 50:  # Less than 50% stability
                    health_status['needs_attention'] = True
                    health_status['reasons'].append(f"Low pattern stability: {stability}%")
            
            # 2. Check recent accuracy
            recent_performance = self.analyze_recent_performance()
            if recent_performance['average_accuracy'] < 15:  # Less than 15% accuracy
                health_status['needs_attention'] = True
                health_status['reasons'].append("Low prediction accuracy")
            
            # 3. Check pattern changes
            if self.check_significant_pattern_changes():
                health_status['needs_attention'] = True
                health_status['reasons'].append("Significant pattern changes detected")
                
            return health_status
            
        except Exception as e:
            print(f"Error in health check: {e}")
            return {'needs_attention': True, 'reasons': ['Error in health monitoring']}

    def track_prediction_performance(self, prediction, actual_draw):
        """Track prediction performance"""
        try:
            correct_numbers = set(prediction['numbers']).intersection(set(actual_draw))
            accuracy = len(correct_numbers) / len(prediction['numbers'])
            
            performance_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_numbers': prediction['numbers'],
                'actual_numbers': actual_draw,
                'correct_count': len(correct_numbers),
                'accuracy': accuracy,
                'confidence_correlation': self.calculate_confidence_correlation(
                    prediction['numbers'], 
                    prediction['confidence'], 
                    actual_draw
                )
            }
            
            return performance_data
            
        except Exception as e:
            print(f"Error tracking performance: {e}")
            return None

    def save_focused_predictions(self, prediction_data):
        """
        Save focused predictions with confidence scores
        
        Args:
            prediction_data (dict): Dictionary containing prediction data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add to the existing save_to_excel method
            focused_predictions_df = pd.DataFrame({
                'timestamp': [prediction_data['timestamp']],
                'numbers': [str(prediction_data['numbers'])],  # Convert list to string for Excel
                'confidence_scores': [str(prediction_data['confidence'])]  # Convert list to string for Excel
            })
            
            # Save to a new sheet in the existing Excel file
            with pd.ExcelWriter(PATHS['ANALYSIS_RESULTS'], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:  # CORRECT
                focused_predictions_df.to_excel(writer, 
                                             sheet_name='Focused_Predictions', 
                                             index=False)
            return True
        except Exception as e:
            print(f"Error saving focused predictions: {e}")
            traceback.print_exc()
            return False

    def analyze_recent_performance(self, window_size=24):
        """
        Analyze recent prediction performance
        
        Args:
            window_size (int): Number of recent draws to analyze
            
        Returns:
            dict: Dictionary containing accuracy metrics
        """
        try:
            recent_draws = self.draws[-window_size:] if len(self.draws) > window_size else self.draws
            # Calculate performance metrics
            total_accuracy = 0
            predictions_count = 0
            
            for i in range(len(recent_draws) - 1):
                prediction = self.get_focused_prediction()
                if prediction:
                    actual_draw = recent_draws[i + 1][1]
                    correct = len(set(prediction['numbers']).intersection(set(actual_draw)))
                    total_accuracy += correct / len(prediction['numbers'])
                    predictions_count += 1
            
            return {
                'average_accuracy': round(total_accuracy / predictions_count if predictions_count > 0 else 0, 4),
                'predictions_analyzed': predictions_count
            }
        except Exception as e:
            print(f"Error in recent performance analysis: {e}")
            return {'average_accuracy': 0, 'predictions_analyzed': 0}

    def calculate_confidence_correlation(self, predicted_numbers, confidence_scores, actual_numbers):
        """
        Calculate correlation between confidence scores and actual hits
        
        Args:
            predicted_numbers (list): List of predicted numbers
            confidence_scores (list): List of confidence scores for each prediction
            actual_numbers (list): List of numbers that actually appeared
            
        Returns:
            float: Correlation coefficient between confidence and actual hits
        """
        try:
            hits = [1 if num in actual_numbers else 0 for num in predicted_numbers]
            if len(hits) != len(confidence_scores):
                return 0
                
            # Calculate correlation coefficient
            n = len(hits)
            hits_mean = sum(hits) / n
            conf_mean = sum(confidence_scores) / n
            
            numerator = sum((h - hits_mean) * (c - conf_mean) 
                           for h, c in zip(hits, confidence_scores))
            denominator = (sum((h - hits_mean) ** 2 for h in hits) * 
                          sum((c - conf_mean) ** 2 for c in confidence_scores)) ** 0.5
            
            correlation = numerator / denominator if denominator != 0 else 0
            return round(correlation, 4)
            
        except Exception as e:
            print(f"Error calculating confidence correlation: {e}")
            return 0

    def check_significant_pattern_changes(self, threshold=0.30):
        """
        Check for significant changes in number patterns
        
        Args:
            threshold (float): Threshold for determining significant change
            
        Returns:
            bool: True if significant pattern changes detected, False otherwise
        """
        try:
            # Compare recent vs historical patterns
            window_size = min(50, len(self.draws) // 2)
            recent_draws = self.draws[-window_size:]
            historical_draws = self.draws[:-window_size]
            
            if not historical_draws:
                return False
                
            # Analyze both sets
            recent_analysis = DataAnalysis(recent_draws, debug=False)
            historical_analysis = DataAnalysis(historical_draws, debug=False)
            
            # Compare frequency distributions
            recent_freq = recent_analysis.count_frequency()
            historical_freq = historical_analysis.count_frequency()
            
            # Calculate pattern change magnitude
            total_change = 0
            total_numbers = 0
            
            for num in range(1, 81):
                recent_prob = recent_freq.get(num, 0) / len(recent_draws) if recent_draws else 0
                hist_prob = historical_freq.get(num, 0) / len(historical_draws) if historical_draws else 0
                
                if hist_prob > 0:  # Only consider numbers that appeared in historical data
                    change = abs(recent_prob - hist_prob) / hist_prob
                    total_change += change
                    total_numbers += 1
            
            avg_change = total_change / total_numbers if total_numbers > 0 else 0
            return avg_change > threshold
            
        except Exception as e:
            print(f"Error checking pattern changes: {e}")
            return False

    def analyze_method_success(self, predictions_df):
        """Analyze success rates of different prediction methods"""
        method_success = {
            'pattern_based': 0.0,
            'frequency_based': 0.0,
            'gap_based': 0.0,
            'combination_based': 0.0
        }
        
        try:
            # Calculate each method's success rate from predictions_df
            for _, prediction in predictions_df.iterrows():
                predicted = prediction['predicted_numbers']
                actual = prediction['actual_numbers']
                
                # Calculate hit rates for each method
                pattern_hits = len(set(predicted) & set(actual)) / len(predicted)
                method_success['pattern_based'] += pattern_hits
                
                # Get frequency-based hits
                freq_numbers = self.get_top_numbers(15)  # Same length as predictions
                freq_hits = len(set(freq_numbers) & set(actual)) / len(freq_numbers)
                method_success['frequency_based'] += freq_hits
                
                # Get gap-based hits
                gaps = self.analyze_gaps()
                gap_numbers = sorted(gaps.items(), key=lambda x: x[1]['current_gap'], reverse=True)[:15]
                gap_hits = len(set(n[0] for n in gap_numbers) & set(actual)) / 15
                method_success['gap_based'] += gap_hits
                
                # Get combination-based hits
                combos = self.analyze_combinations()['most_common'][:5]
                combo_numbers = {num for combo in combos for num in combo['combination']}
                combo_numbers = list(combo_numbers)[:15]  # Take top 15
                combo_hits = len(set(combo_numbers) & set(actual)) / len(combo_numbers)
                method_success['combination_based'] += combo_hits
                
            # Average the success rates
            num_predictions = len(predictions_df)
            for method in method_success:
                method_success[method] /= num_predictions
                
            return method_success
            
        except Exception as e:
            print(f"Error in analyze_method_success: {e}")
            return method_success

    def analyze_accuracy_trend(self, predictions_df):
        """Analyze prediction accuracy trends over time"""
        try:
            trend_data = {
                'is_improving': False,
                'window_accuracy': [],
                'overall_trend': 0.0
            }
            
            # Calculate rolling accuracy
            window_size = 5
            accuracies = []
            
            for i in range(len(predictions_df)):
                if i < window_size:
                    continue
                    
                window = predictions_df.iloc[i-window_size:i]
                window_hits = 0
                window_total = 0
                
                for _, prediction in window.iterrows():
                    predicted = prediction['predicted_numbers']
                    actual = prediction['actual_numbers']
                    hits = len(set(predicted) & set(actual))
                    window_hits += hits
                    window_total += len(predicted)
                
                window_accuracy = window_hits / window_total if window_total > 0 else 0
                accuracies.append(window_accuracy)
                
            if accuracies:
                # Calculate trend
                trend_data['is_improving'] = accuracies[-1] > accuracies[0]
                trend_data['window_accuracy'] = accuracies
                trend_data['overall_trend'] = (accuracies[-1] - accuracies[0]) / len(accuracies)
                
            return trend_data
            
        except Exception as e:
            print(f"Error in analyze_accuracy_trend: {e}")
            return {'is_improving': False, 'window_accuracy': [], 'overall_trend': 0.0}

    def analyze_prediction_history_files(self):
        """Analyze prediction history files to learn from past predictions"""
        print("\n=== Analyzing Prediction History Files ===")
        
        try:
            # Load prediction history
            prediction_file = os.path.join(PATHS['PREDICTIONS_DIR'], 'prediction_history.pkl')
            evaluation_file = os.path.join(PATHS['PREDICTIONS_DIR'], 'prediction_evaluations.pkl')
            
            if not os.path.exists(prediction_file):
                print("No prediction history file found")
                return None
                
            predictions_df = pd.read_pickle(prediction_file)
            evaluations_df = pd.read_pickle(evaluation_file) if os.path.exists(evaluation_file) else None
            
            print(f"Loaded {len(predictions_df)} previous predictions")
            
            # Analyze which methods were most successful
            method_success = self.analyze_method_success(predictions_df)
            
            # Get recent accuracy trends
            accuracy_trend = self.analyze_accuracy_trend(predictions_df)
            
            # Check pattern stability
            pattern_changes = self.check_significant_pattern_changes()
            
            results = {
                'method_success': method_success,
                'accuracy_trend': accuracy_trend,
                'pattern_stability': not pattern_changes,
                'recommendations': {
                    'best_method': max(method_success.items(), key=lambda x: x[1])[0] if method_success else None,
                    'accuracy_improving': accuracy_trend['is_improving'],
                    'needs_retraining': pattern_changes
                }
            }
            
            print("\n=== Learning from Previous Predictions ===")
            print(f"Best performing method: {results['recommendations']['best_method']}")
            print(f"Accuracy trend: {'Improving' if results['recommendations']['accuracy_improving'] else 'Declining'}")
            print(f"Retraining needed: {'Yes' if results['recommendations']['needs_retraining'] else 'No'}")
            
            return results
            
        except Exception as e:
            print(f"Error analyzing prediction history: {e}")
            traceback.print_exc()
            return None

    def update_prediction_tracker(self, predicted_numbers, actual_numbers, correct_count, accuracy):
        """
        Export prediction results to Excel without modifying core functionality
        This function only exports data that's already calculated by the system
        
        Args:
            predicted_numbers (list): List of predicted numbers
            actual_numbers (list): List of actual draw numbers  
            correct_count (int): Number of correct predictions
            accuracy (float): Hit rate (correct_count / len(predicted_numbers))
        
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            import pandas as pd
            from datetime import datetime
            
            # Path to tracker file
            tracker_file = r"C:\Users\MihaiNita\OneDrive - Prime Batteries\Desktop\versiuni_de_care_nu_ma_ating\Versiune1.4\data\processed\prediction_tracker.xlsx"
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M')
            
            # Create or load existing tracker
            if os.path.exists(tracker_file):
                try:
                    df = pd.read_excel(tracker_file)
                except Exception as e:
                    print(f"Could not read existing tracker file: {e}")
                    df = pd.DataFrame(columns=[
                        'Date', 'Time', 'Predicted_Numbers', 'Actual_Numbers',
                        'Hits', 'Hit_Rate', 'Notes'
                    ])
            else:
                df = pd.DataFrame(columns=[
                    'Date', 'Time', 'Predicted_Numbers', 'Actual_Numbers',
                    'Hits', 'Hit_Rate', 'Notes'
                ])
            
            # Create new entry
            new_entry = {
                'Date': current_date,
                'Time': current_time,
                'Predicted_Numbers': str(predicted_numbers),
                'Actual_Numbers': str(actual_numbers),
                'Hits': correct_count,
                'Hit_Rate': accuracy,
                'Notes': f"Auto-tracked on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }
            
            # Add to dataframe
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Save to Excel
            try:
                df.to_excel(tracker_file, index=False)
                print(f"✅ Prediction tracker updated: {tracker_file}")
                if self.debug:
                    print(f"DEBUG: Added prediction entry with {correct_count} hits and {accuracy:.2%} accuracy")
                return True
            except Exception as e:
                print(f"❌ Could not update prediction tracker: {e}")
                return False
                
        except Exception as e:
            print(f"Error in prediction tracker: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    example_draws = [
        (datetime.now().strftime("%H:%M %d-%m-%Y"), [1, 2, 2, 9, 12, 14, 17, 25, 26, 30, 38, 44, 54, 57, 58, 61, 65, 71, 72, 76, 79]),
        ((datetime.now() - timedelta(minutes=5)).strftime("%H:%M %d-%m-%Y"), [4, 5, 7, 7, 9, 18, 24, 27, 29, 34, 40, 45, 48, 52, 55, 57, 70, 71, 72, 74, 77]),
    ]
    print_system_info()
    try:
        # Initialize real draws list and get the path to historical data
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
            
        # Create analysis instance with our data
        analysis = DataAnalysis(draws_to_analyze)
        
        # Basic frequency analysis
        frequency = analysis.count_frequency()
        print(f"Number of unique numbers: {len(frequency)}")
        
        # Top numbers analysis
        top_numbers = analysis.get_top_numbers(20)
        print(f"Top 20 numbers: {', '.join(map(str, top_numbers))}")
        
        # Gap analysis showcase
        gap_analysis = analysis.analyze_gaps()
        print("\nSample gap analysis for first 5 numbers:")
        for num in range(1, 6):
            print(f"Number {num}: {gap_analysis[num]}")
        
        # Focused prediction showcase
        focused_prediction = analysis.get_focused_prediction()
        if focused_prediction:
            print("\nFocused Prediction (Top 15 numbers):")
            for num, conf in zip(focused_prediction['numbers'], 
                                focused_prediction['confidence']):
                print(f"Number {num}: {conf:.3f} confidence")
        
        # Save results to Excel
        analysis.save_to_excel(PATHS['ANALYSIS_RESULTS'])
        print("Analysis complete!")
        
        # Deep Pattern Analysis Section
        print("\n=== Starting Deep Pattern Analysis ===")
        
        print("\n1. Analyzing Sequence Patterns...")
        sequence_analysis = analysis.sequence_pattern_analysis()
        if sequence_analysis and 'overall_sequences' in sequence_analysis:
            print(f"Found patterns in {len(sequence_analysis['overall_sequences'])} sequences")
            if 'statistics' in sequence_analysis and 'most_active_time_slot' in sequence_analysis['statistics']:
                print(f"Most active time: {sequence_analysis['statistics']['most_active_time_slot']}")
        
        print("\n2. Analyzing Skip Patterns...")
        skip_patterns = analysis.analyze_skip_patterns()
        if skip_patterns and 'statistics' in skip_patterns:
            print(f"Average skipped numbers: {skip_patterns['statistics'].get('avg_skipped', 0):.2f}")
            print(f"Average new numbers: {skip_patterns['statistics'].get('avg_new', 0):.2f}")
        
        print("\n3. Checking System Health...")
        health_status = analysis.check_prediction_health()
        if health_status:
            if health_status.get('needs_attention'):
                print("System needs attention:")
                for reason in health_status.get('reasons', []):
                    print(f"- {reason}")
            else:
                print("System health: Good")
        
        print("\n4. Validating Patterns...")
        pattern_validation = analysis.validate_patterns()
        if pattern_validation and 'validation_summary' in pattern_validation:
            summary = pattern_validation['validation_summary']
            print(f"Pattern Stability: {summary.get('predictive_power', 'N/A')}")
            print(f"Hit Rate vs Random: {summary.get('hit_rate_advantage', 0):.2f}")
        
        print("\n=== Deep Analysis Complete ===")
        
        # Performance Analysis Section
        print("\n=== Starting Performance Analysis ===")
        
        print("\nAnalyzing recent performance...")
        recent_performance = analysis.analyze_recent_performance()
        if recent_performance:
            print(f"Average Accuracy: {recent_performance['average_accuracy']*100:.2f}%")
            print(f"Number of Predictions Analyzed: {recent_performance['predictions_analyzed']}")
        
        print("\nSaving current focused prediction...")
        if focused_prediction:
            saved = analysis.save_focused_predictions(focused_prediction)
            if saved:
                print("Successfully saved focused prediction")
            else:
                print("Failed to save focused prediction")
        
        print("\nChecking prediction performance...")
        # Get most recent actual draw for comparison
        if draws_to_analyze and len(draws_to_analyze) > 0:
            latest_actual_draw = draws_to_analyze[-1][1]  # Get numbers from last draw
            if focused_prediction and latest_actual_draw:
                performance_data = analysis.track_prediction_performance(
                    focused_prediction,
                    latest_actual_draw
                )
                if performance_data:
                    print(f"Latest Prediction Accuracy: {performance_data['accuracy']*100:.2f}%")
                    print(f"Correct Numbers: {performance_data['correct_count']}")
                    print(f"Confidence Correlation: {performance_data['confidence_correlation']:.4f}")
                    analysis.update_prediction_tracker(
                    predicted_numbers=focused_prediction['numbers'],
                    actual_numbers=latest_actual_draw,
                    correct_count=performance_data['correct_count'],
                    accuracy=performance_data['accuracy'] 
                    )   
        print("\n=== Analysis and Performance Tracking Complete ===")
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
        traceback.print_exc()