from collections import Counter, defaultdict
from datetime import datetime
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
            print(f"DEBUG: First draw format: {draws[0] if draws else 'None'}")
        
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
    
    def hot_and_cold_numbers(self, top_n=10, window_size=50):
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
            filename = PATHS['ANALYSIS']
        
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
                    'Combination': str(list(item['combination'])).replace('[', '').replace(']', ''),  # More readable format
                    'Frequency': item['frequency'],
                    'Percentage': round(item['percentage'], 2),
                    'Average_Gap': round(item['average_gap'], 2)
                }
                for item in combinations_analysis['most_common']
            ])
            
            least_common_combinations_df = pd.DataFrame([
                {
                    'Combination': str(list(item['combination'])).replace('[', '').replace(']', ''),  # More readable format
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

    def analyze_skip_patterns(self, window_size=10):
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

if __name__ == "__main__":
    example_draws = [
        ("20:15 26-02-2025", [1, 2, 2, 9, 12, 14, 17, 25, 26, 30, 38, 44, 54, 57, 58, 61, 65, 71, 72, 76, 79]),
        ("20:10 26-02-2025", [4, 5, 7, 7, 9, 18, 24, 27, 29, 34, 40, 45, 48, 52, 55, 57, 70, 71, 72, 74, 77]),
    ]
    
    try:
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
            
        analysis = DataAnalysis(draws_to_analyze)
        
        frequency = analysis.count_frequency()
        print(f"Number of unique numbers: {len(frequency)}")
        
        top_numbers = analysis.get_top_numbers(20)
        print(f"Top 20 numbers: {', '.join(map(str, top_numbers))}")
        
        # Add gap analysis showcase
        gap_analysis = analysis.analyze_gaps()
        print("\nSample gap analysis for first 5 numbers:")
        for num in range(1, 6):
            print(f"Number {num}: {gap_analysis[num]}")
        
        analysis.save_to_excel(PATHS['ANALYSIS'])
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
        traceback.print_exc()