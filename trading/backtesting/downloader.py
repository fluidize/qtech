import sys
sys.path.append("")

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QComboBox, QPushButton, QTextEdit, QProgressBar,
                             QMessageBox, QGroupBox, QFrame, QListWidget, QListWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QFont, QPalette, QColor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import trading.model_tools as mt

class SingleDownloadWorker(QThread):
    progress_update = pyqtSignal(str, str)
    download_complete = pyqtSignal(str, object, str)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, symbol, days, interval, age_days, data_source):
        super().__init__()
        self.symbol = symbol
        self.days = days
        self.interval = interval
        self.age_days = age_days
        self.data_source = data_source
        self.symbol_interval = f"{symbol}-{interval}"
        
    def run(self):
        try:
            self.progress_update.emit(self.symbol_interval, f"Starting download for {self.symbol} ({self.interval})")
            self.progress_update.emit(self.symbol_interval, f"Parameters: {self.days} days, {self.interval} interval, {self.age_days} age days, {self.data_source} source")
            
            data = mt.fetch_data(
                symbol=self.symbol,
                days=self.days,
                interval=self.interval,
                age_days=self.age_days,
                data_source=self.data_source,
                use_cache=True,
                cache_expiry_hours=24,
                retry_limit=5,
                verbose=True
            )
            
            self.progress_update.emit(self.symbol_interval, "Download completed successfully!")
            self.progress_update.emit(self.symbol_interval, f"Data shape: {data.shape}")
            self.progress_update.emit(self.symbol_interval, f"Columns: {list(data.columns)}")
            
            if not data.empty:
                self.progress_update.emit(self.symbol_interval, f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
                self.download_complete.emit(self.symbol_interval, data, "Download completed successfully!")
            else:
                self.progress_update.emit(self.symbol_interval, "Warning: No data was downloaded")
                self.download_complete.emit(self.symbol_interval, data, "No data downloaded")
                
        except Exception as e:
            error_msg = f"Error during download: {str(e)}"
            self.progress_update.emit(self.symbol_interval, error_msg)
            self.error_occurred.emit(self.symbol_interval, error_msg)

class MultiDownloadManager(QThread):
    progress_update = pyqtSignal(str, str)
    download_complete = pyqtSignal(str, object, str)
    error_occurred = pyqtSignal(str, str)
    all_downloads_complete = pyqtSignal(int, int)
    def __init__(self, symbols, days, intervals, age_days, data_source, max_workers=5):
        super().__init__()
        self.symbols = symbols
        self.days = days
        self.intervals = intervals
        self.age_days = age_days
        self.data_source = data_source
        self.max_workers = max_workers
        self.completed_count = 0
        self.total_count = len(symbols) * len(intervals)
        self.mutex = QMutex()
        
    def run(self):
        self.progress_update.emit("SYSTEM", f"Starting downloads for {len(self.symbols)} symbols × {len(self.intervals)} intervals = {self.total_count} total downloads with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol_interval = {}
            for symbol in self.symbols:
                for interval in self.intervals:
                    future = executor.submit(self.download_single_symbol_interval, symbol, interval)
                    future_to_symbol_interval[future] = f"{symbol}-{interval}"
            
            for future in as_completed(future_to_symbol_interval):
                symbol_interval = future_to_symbol_interval[future]
                try:
                    data, status = future.result()
                    with QMutexLocker(self.mutex):
                        self.completed_count += 1
                    self.download_complete.emit(symbol_interval, data, status)
                    self.all_downloads_complete.emit(self.completed_count, self.total_count)
                except Exception as e:
                    with QMutexLocker(self.mutex):
                        self.completed_count += 1
                    self.error_occurred.emit(symbol_interval, str(e))
                    self.all_downloads_complete.emit(self.completed_count, self.total_count)
        
        self.progress_update.emit("SYSTEM", f"All downloads completed! {self.completed_count}/{self.total_count} successful")
    
    def download_single_symbol_interval(self, symbol, interval):
        symbol_interval = f"{symbol}-{interval}"
        try:
            self.progress_update.emit(symbol_interval, f"Starting download for {symbol} ({interval})")
            
            data = mt.fetch_data(
                symbol=symbol,
                days=self.days,
                interval=interval,
                age_days=self.age_days,
                data_source=self.data_source,
                use_cache=True,
                cache_expiry_hours=24,
                retry_limit=5,
                verbose=True
            )
            
            self.progress_update.emit(symbol_interval, "Download completed successfully!")
            self.progress_update.emit(symbol_interval, f"Data shape: {data.shape}")
            
            if not data.empty:
                self.progress_update.emit(symbol_interval, f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
                return data, "Download completed successfully!"
            else:
                self.progress_update.emit(symbol_interval, "Warning: No data was downloaded")
                return data, "No data downloaded"
                
        except Exception as e:
            error_msg = f"Error during download: {str(e)}"
            self.progress_update.emit(symbol_interval, error_msg)
            raise e

class DataDownloaderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.download_manager = None
        self.downloaded_data = {}
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Market Data Downloader - PyQt")
        self.setGeometry(100, 100, 700, 600)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit, QComboBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
                font-size: 12px;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = QLabel("Market Data Downloader")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        input_group = QGroupBox("Download Parameters")
        input_layout = QGridLayout(input_group)
        input_layout.setSpacing(10)
        
        input_layout.addWidget(QLabel("Symbols:"), 0, 0)
        self.symbol_edit = QLineEdit("SOL-USDT")
        self.symbol_edit.setPlaceholderText("e.g., BTC-USDT, ETH-USDT, ADA-USDT (comma-separated)")
        input_layout.addWidget(self.symbol_edit, 0, 1)
        
        input_layout.addWidget(QLabel("Max Workers:"), 1, 0)
        self.max_workers_edit = QLineEdit("5")
        self.max_workers_edit.setPlaceholderText("Number of parallel downloads (1-10)")
        input_layout.addWidget(self.max_workers_edit, 1, 1)
        
        input_layout.addWidget(QLabel("Days:"), 2, 0)
        self.days_edit = QLineEdit("7")
        self.days_edit.setPlaceholderText("Number of days to download")
        input_layout.addWidget(self.days_edit, 2, 1)
        
        input_layout.addWidget(QLabel("Intervals:"), 3, 0)
        self.interval_edit = QLineEdit("15m, 30m, 1h, 4h")
        self.interval_edit.setPlaceholderText("e.g., 1m, 5m, 1h, 1d (comma-separated)")
        input_layout.addWidget(self.interval_edit, 3, 1)
        
        input_layout.addWidget(QLabel("Age Days:"), 4, 0)
        self.age_days_edit = QLineEdit("0")
        self.age_days_edit.setPlaceholderText("Days back to start from (0 = recent)")
        input_layout.addWidget(self.age_days_edit, 4, 1)
        
        input_layout.addWidget(QLabel("Data Source:"), 5, 0)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["binance", "kucoin", "yfinance"])
        self.data_source_combo.setCurrentText("binance")
        input_layout.addWidget(self.data_source_combo, 5, 1)
        
        main_layout.addWidget(input_group)
        
        self.download_button = QPushButton("Download Data")
        self.download_button.clicked.connect(self.start_download)
        self.download_button.setMinimumHeight(40)
        main_layout.addWidget(self.download_button)
        
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_label = QLabel("Ready to download...")
        self.progress_label.setStyleSheet("color: #666; font-style: italic;")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.symbol_progress_list = QListWidget()
        self.symbol_progress_list.setMaximumHeight(150)
        self.symbol_progress_list.setVisible(False)
        progress_layout.addWidget(self.symbol_progress_list)
        
        main_layout.addWidget(progress_group)
        
        status_group = QGroupBox("Download Status & Results")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(200)
        
        help_text = """Help:
• Symbols: Trading pairs separated by commas (e.g., BTC-USDT, ETH-USDT, ADA-USDT)
• Max Workers: Number of parallel downloads (1-10, default: 5)
• Days: Number of days of data to download
• Intervals: Time intervals separated by commas (e.g., 1m, 5m, 1h, 1d)
• Age Days: How many days back to start from (0 = most recent)
• Data Source: Choose from binance, kucoin, or yfinance

Data will be cached automatically to avoid unnecessary downloads.
Cache expires after 24 hours. Retry limit is set to 5 attempts.
Multiple symbols and intervals will be downloaded simultaneously for faster processing.
        """
        self.status_text.setPlainText(help_text)
        
        status_layout.addWidget(self.status_text)
        main_layout.addWidget(status_group)
        
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)
        main_layout.setStretch(4, 1)
        
    def log_message(self, symbol, message):
        if symbol == "SYSTEM":
            self.status_text.append(f"[SYSTEM] {message}")
        else:
            self.status_text.append(f"[{symbol}] {message}")
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)
        
    def update_symbol_progress(self, symbol_interval, status):
        items = [self.symbol_progress_list.item(i) for i in range(self.symbol_progress_list.count())]
        existing_item = None
        for item in items:
            if item.text().startswith(f"{symbol_interval}:"):
                existing_item = item
                break
        
        if existing_item:
            existing_item.setText(f"{symbol_interval}: {status}")
        else:
            new_item = QListWidgetItem(f"{symbol_interval}: {status}")
            self.symbol_progress_list.addItem(new_item)
        
    def start_download(self):
        try:
            symbols_text = self.symbol_edit.text().strip()
            max_workers = int(self.max_workers_edit.text())
            days = int(self.days_edit.text())
            intervals_text = self.interval_edit.text().strip()
            age_days = int(self.age_days_edit.text())
            data_source = self.data_source_combo.currentText()
            
            if not symbols_text:
                QMessageBox.warning(self, "Error", "Please enter at least one symbol")
                return
            
            if not intervals_text:
                QMessageBox.warning(self, "Error", "Please enter at least one interval")
                return
            
            symbols = [s.strip() for s in symbols_text.split(',') if s.strip()]
            if not symbols:
                QMessageBox.warning(self, "Error", "Please enter valid symbols")
                return
            
            # Parse intervals
            intervals = [i.strip() for i in intervals_text.split(',') if i.strip()]
            if not intervals:
                QMessageBox.warning(self, "Error", "Please enter valid intervals")
                return
                
            if max_workers < 1 or max_workers > 10:
                QMessageBox.warning(self, "Error", "Max workers must be between 1 and 10")
                return
                
            if days <= 0:
                QMessageBox.warning(self, "Error", "Days must be a positive number")
                return
                
            if age_days < 0:
                QMessageBox.warning(self, "Error", "Age days cannot be negative")
                return
                
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")
            return
        
        # Disable download button and show progress
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        self.progress_bar.setVisible(True)
        total_downloads = len(symbols) * len(intervals)
        self.progress_bar.setRange(0, total_downloads)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Downloading {len(symbols)} symbols × {len(intervals)} intervals = {total_downloads} total downloads...")
        
        self.symbol_progress_list.setVisible(True)
        self.symbol_progress_list.clear()
        
        self.status_text.clear()
        self.downloaded_data.clear()
        
        # Create and start download manager
        self.download_manager = MultiDownloadManager(symbols, days, intervals, age_days, data_source, max_workers)
        self.download_manager.progress_update.connect(self.log_message)
        self.download_manager.download_complete.connect(self.download_complete)
        self.download_manager.error_occurred.connect(self.download_error)
        self.download_manager.all_downloads_complete.connect(self.update_overall_progress)
        self.download_manager.start()
        
    def download_complete(self, symbol_interval, data, status_message):
        """Handle successful download completion for a single symbol-interval"""
        self.downloaded_data[symbol_interval] = data
        self.update_symbol_progress(symbol_interval, "✓ Completed")
        
    def download_error(self, symbol_interval, error_message):
        """Handle download error for a single symbol-interval"""
        self.update_symbol_progress(symbol_interval, f"✗ Error: {error_message}")
        
    def update_overall_progress(self, completed, total):
        """Update overall progress bar"""
        self.progress_bar.setValue(completed)
        self.progress_label.setText(f"Downloaded {completed}/{total} symbol-interval combinations...")
        
        if completed >= total:
            self.download_button.setEnabled(True)
            self.download_button.setText("Download Data")
            self.progress_label.setText(f"All downloads completed! {len(self.downloaded_data)}/{total} successful")
            self.progress_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            
            QTimer.singleShot(5000, lambda: self.symbol_progress_list.setVisible(False))
            
            QTimer.singleShot(3000, lambda: self.progress_label.setStyleSheet("color: #666; font-style: italic;"))
            
            successful = len(self.downloaded_data)
            failed = total - successful
            if failed > 0:
                QMessageBox.warning(self, "Download Summary", 
                                  f"Downloads completed!\n\nSuccessful: {successful}\nFailed: {failed}\n\nCheck the log for details.")
            else:
                QMessageBox.information(self, "Download Complete", 
                                      f"All {successful} downloads completed successfully!")

def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("Market Data Downloader")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("")
    
    window = DataDownloaderApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
