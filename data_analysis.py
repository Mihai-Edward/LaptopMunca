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


    def find_common_pairs(self, top_n=30):
        """Find most common pairs of numbers"""
        pairs = Counter()
        for draw in self.draws:
            numbers = draw[1]
            pairs.update(combinations(sorted(numbers), 2))
        common_pairs = pairs.most_common(top_n)
        print(f"DEBUG: Found {len(common_pairs)} common pairs")
        return common_pairs

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
            
            if not os.path.exists(PATHS['ANALYSIS_RESULTS']):
                with pd.ExcelWriter(PATHS['ANALYSIS_RESULTS'], engine='openpyxl') as writer:
                    pass  # Create an empty Excel file

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
            combinations_analysis = self.analyze_combinations(group_size=3, top_n=30)
            
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

    def analyze_combinations(self, group_size=3, top_n=30):
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
        
        # Common pairs analysis
        common_pairs = analysis.find_common_pairs(30)
        print("\nMost common pairs:")
        for pair, freq in common_pairs[:5]:
            print(f"Pair {pair}: appeared {freq} times")
            
        # Hot and cold numbers analysis
        hot_cold = analysis.hot_and_cold_numbers()
        print("\nHot/Cold Analysis:")
        print("Top 5 Hot Numbers:", [num for num, _ in hot_cold['hot_numbers'][:5]])
        print("Top 5 Cold Numbers:", [num for num, _ in hot_cold['cold_numbers'][:5]])
        
        # Gap analysis
        gap_analysis = analysis.analyze_gaps()
        print("\nGap Analysis for first 5 numbers:")
        for num in range(1, 6):
            gap_data = gap_analysis[num]
            print(f"Number {num}: Average gap {gap_data['avg_gap']}, Current gap {gap_data['current_gap']}")
        
        # Combinations analysis
        combinations = analysis.analyze_combinations(group_size=3, top_n=10)
        print("\nTop 5 Common Combinations:")
        for combo in combinations['most_common'][:5]:
            print(f"Combination {combo['combination']}: Frequency {combo['frequency']}")
        
        # Skip patterns analysis
        skip_patterns = analysis.analyze_skip_patterns()
        if 'statistics' in skip_patterns:
            print(f"\nAverage numbers skipped per draw: {skip_patterns['statistics']['avg_skipped']:.2f}")
            print(f"Average new numbers per draw: {skip_patterns['statistics']['avg_new']:.2f}")
        
        # Save all analysis results to Excel
        analysis.save_to_excel(PATHS['ANALYSIS_RESULTS'])
        print("\nAnalysis results saved to Excel!")
        
        print("\n=== Analysis Complete ===")
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
        traceback.print_exc()