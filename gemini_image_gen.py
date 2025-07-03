import sys
import os
import json
import base64
from io import BytesIO
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLineEdit, QTextEdit, QLabel, 
                             QFileDialog, QMessageBox, QStatusBar, QGroupBox, QFormLayout,
                             QFrame, QSizePolicy, QSpacerItem, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QDragEnterEvent, QDropEvent
import qtawesome as qta

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class DropLabel(QLabel):
    """Label that accepts drag and drop for image files"""
    file_dropped = Signal(str)
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)
            event.acceptProposedAction()

class ImageWorker(QThread):
    """Worker thread for both image generation and recognition"""
    image_generated = Signal(object)
    image_recognized = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, api_key, operation, data):
        super().__init__()
        self.api_key = api_key
        self.operation = operation  # 'generate' or 'recognize'
        self.data = data
    
    def run(self):
        try:
            if not GEMINI_AVAILABLE:
                if self.operation == 'generate':
                    self.generate_mock_image()
                else:
                    self.error_occurred.emit("Image recognition requires Gemini API")
                return
            
            client = genai.Client(api_key=self.api_key)
            
            if self.operation == 'generate':
                self.generate_image(client)
            elif self.operation == 'recognize':
                self.recognize_image(client)
                
        except Exception as e:
            self.error_occurred.emit(f"API Error: {str(e)}")
    
    def generate_image(self, client):
        """Generate image from prompt"""
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=self.data,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                self.image_generated.emit(image)
                return
        
        self.error_occurred.emit("No image found in API response")
    
    def recognize_image(self, client):
        """Recognize and describe image"""
        with open(self.data, 'rb') as f:
            image_bytes = f.read()
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg' if self.data.lower().endswith('.jpg') or self.data.lower().endswith('.jpeg') else 'image/png',
                ),
                'Describe this image in detail for AI image generation purposes. Focus on visual elements, style, composition, colors, and mood.'
            ]
        )
        
        self.image_recognized.emit(response.text)
    
    def generate_mock_image(self):
        """Generate a placeholder image for testing"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (400, 400), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            text = "Mock Image\n\n" + self.data[:80] + "..."
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            lines = text.split('\n')
            y_offset = 40
            for line in lines:
                draw.text((15, y_offset), line, fill='darkblue', font=font)
                y_offset += 25
            
            self.image_generated.emit(img)
            
        except Exception as e:
            self.error_occurred.emit(f"Mock image generation failed: {str(e)}")

class GeminiImageGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.last_prompt = ""
        self.config_file = "config.json"
        self.dark_theme = True
        
        self.init_ui()
        self.load_config()
        self.apply_theme()
        
        self.status_timer = QTimer()
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.clear_status)
    
    def init_ui(self):
        """Initialize the compact user interface"""
        self.setWindowTitle("Gemini Image Generator")
        self.setFixedSize(480, 720)
        self.setWindowIcon(qta.icon('fa5s.magic', color='#6366f1'))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        
        # Compact header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(2)
        
        title_label = QLabel("Gemini Image Gen")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("themeButton")
        self.theme_btn.setFixedSize(30, 30)
        self.theme_btn.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_btn)
        
        main_layout.addLayout(header_layout)
        
        # API Key (compact)
        api_layout = QHBoxLayout()
        api_layout.setSpacing(2)
        api_label = QLabel("API:")
        api_label.setFixedWidth(30)
        api_layout.addWidget(api_label)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setObjectName("compactInput")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Gemini API key...")
        self.api_key_input.textChanged.connect(self.save_config)
        api_layout.addWidget(self.api_key_input)
        
        main_layout.addLayout(api_layout)
        
        # Context section (drag & drop)
        context_frame = QFrame()
        context_frame.setObjectName("compactFrame")
        context_layout = QVBoxLayout(context_frame)
        context_layout.setContentsMargins(4, 4, 4, 4)
        context_layout.setSpacing(2)
        
        context_header = QHBoxLayout()
        context_header.setSpacing(2)
        context_title = QLabel("Context")
        context_title.setObjectName("sectionTitle")
        context_header.addWidget(context_title)
        
        self.identify_btn = QPushButton("Browse")
        self.identify_btn.setObjectName("smallButton")
        self.identify_btn.setFixedSize(50, 20)
        self.identify_btn.clicked.connect(self.browse_image)
        context_header.addWidget(self.identify_btn)
        
        context_layout.addLayout(context_header)
        
        self.drop_area = DropLabel("📁 Drop image here or browse")
        self.drop_area.setObjectName("dropArea")
        self.drop_area.setFixedHeight(60)
        self.drop_area.file_dropped.connect(self.handle_dropped_file)
        context_layout.addWidget(self.drop_area)
        
        self.context_output = QTextEdit()
        self.context_output.setObjectName("compactTextEdit")
        self.context_output.setMaximumHeight(80)
        self.context_output.setPlaceholderText("Image description will appear here...")
        context_layout.addWidget(self.context_output)
        
        main_layout.addWidget(context_frame)
        
        # Prompt section
        prompt_frame = QFrame()
        prompt_frame.setObjectName("compactFrame")
        prompt_layout = QVBoxLayout(prompt_frame)
        prompt_layout.setContentsMargins(4, 4, 4, 4)
        prompt_layout.setSpacing(2)
        
        prompt_title = QLabel("Prompt")
        prompt_title.setObjectName("sectionTitle")
        prompt_layout.addWidget(prompt_title)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setObjectName("compactTextEdit")
        self.prompt_input.setMaximumHeight(80)
        self.prompt_input.setPlaceholderText("Describe your image...")
        prompt_layout.addWidget(self.prompt_input)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(2)
        
        self.use_context_btn = QPushButton("Use Context")
        self.use_context_btn.setObjectName("smallButton")
        self.use_context_btn.setFixedHeight(25)
        self.use_context_btn.clicked.connect(self.use_context)
        self.use_context_btn.setEnabled(False)
        
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.setFixedHeight(25)
        self.generate_btn.clicked.connect(self.generate_image)
        
        self.regenerate_btn = QPushButton("Regen")
        self.regenerate_btn.setObjectName("secondaryButton")
        self.regenerate_btn.setFixedHeight(25)
        self.regenerate_btn.clicked.connect(self.regenerate_image)
        self.regenerate_btn.setEnabled(False)
        
        button_layout.addWidget(self.use_context_btn)
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.regenerate_btn)
        
        prompt_layout.addLayout(button_layout)
        main_layout.addWidget(prompt_frame)
        
        # Image display
        image_frame = QFrame()
        image_frame.setObjectName("compactFrame")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(4, 4, 4, 4)
        image_layout.setSpacing(2)
        
        image_header = QHBoxLayout()
        image_header.setSpacing(2)
        image_title = QLabel("Result")
        image_title.setObjectName("sectionTitle")
        image_header.addWidget(image_title)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("smallButton")
        self.save_btn.setFixedSize(40, 20)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        image_header.addWidget(self.save_btn)
        
        image_layout.addLayout(image_header)
        
        self.image_label = DropLabel("✨ Generated image")
        self.image_label.setObjectName("imageDisplay")
        self.image_label.setFixedSize(468, 300)
        self.image_label.file_dropped.connect(self.handle_dropped_file)
        
        image_layout.addWidget(self.image_label)
        main_layout.addWidget(image_frame)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("compactStatusBar")
        self.setStatusBar(self.status_bar)
        
        if not GEMINI_AVAILABLE:
            self.status_bar.showMessage("⚠️ Mock mode", 3000)
    
    def get_light_theme_style(self):
        """Compact light theme"""
        return """
            QMainWindow { background-color: #f8fafc; color: #1e293b; }
            #titleLabel { font-size: 14px; font-weight: bold; color: #6366f1; }
            #sectionTitle { font-size: 11px; font-weight: 600; color: #374151; }
            #compactFrame { background-color: white; border: 1px solid #e5e7eb; border-radius: 4px; }
            #compactInput, #compactTextEdit { 
                background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 3px; 
                padding: 4px; font-size: 11px; color: #1f2937;
            }
            #compactInput:focus, #compactTextEdit:focus { border-color: #6366f1; background-color: white; }
            #primaryButton { 
                background-color: #6366f1; color: white; border: none; border-radius: 3px; 
                padding: 4px 8px; font-size: 11px; font-weight: 600;
            }
            #primaryButton:hover { background-color: #5b21b6; }
            #primaryButton:disabled { background-color: #9ca3af; }
            #secondaryButton, #smallButton { 
                background-color: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; 
                border-radius: 3px; padding: 4px 8px; font-size: 11px; font-weight: 600;
            }
            #secondaryButton:hover, #smallButton:hover { background-color: #e5e7eb; }
            #secondaryButton:disabled, #smallButton:disabled { background-color: #f9fafb; color: #9ca3af; }
            #themeButton { 
                background-color: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 15px;
            }
            #themeButton:hover { background-color: #e5e7eb; }
            #dropArea { 
                background-color: #f9fafb; border: 2px dashed #d1d5db; border-radius: 4px; 
                color: #6b7280; font-size: 11px;
            }
            #imageDisplay { 
                background-color: #f9fafb; border: 1px solid #d1d5db; border-radius: 4px; 
                color: #6b7280; font-size: 12px;
            }
            #compactStatusBar { background-color: #f8fafc; color: #6b7280; font-size: 10px; }
        """
    
    def get_dark_theme_style(self):
        """Compact dark theme"""
        return """
            QMainWindow { background-color: #0f172a; color: #e2e8f0; }
            #titleLabel { font-size: 14px; font-weight: bold; color: #8b5cf6; }
            #sectionTitle { font-size: 11px; font-weight: 600; color: #cbd5e1; }
            #compactFrame { background-color: #1e293b; border: 1px solid #334155; border-radius: 4px; }
            #compactInput, #compactTextEdit { 
                background-color: #334155; border: 1px solid #475569; border-radius: 3px; 
                padding: 4px; font-size: 11px; color: #f1f5f9;
            }
            #compactInput:focus, #compactTextEdit:focus { border-color: #8b5cf6; background-color: #475569; }
            #primaryButton { 
                background-color: #8b5cf6; color: white; border: none; border-radius: 3px; 
                padding: 4px 8px; font-size: 11px; font-weight: 600;
            }
            #primaryButton:hover { background-color: #7c3aed; }
            #primaryButton:disabled { background-color: #64748b; }
            #secondaryButton, #smallButton { 
                background-color: #475569; color: #e2e8f0; border: 1px solid #64748b; 
                border-radius: 3px; padding: 4px 8px; font-size: 11px; font-weight: 600;
            }
            #secondaryButton:hover, #smallButton:hover { background-color: #64748b; }
            #secondaryButton:disabled, #smallButton:disabled { background-color: #334155; color: #64748b; }
            #themeButton { 
                background-color: #475569; border: 1px solid #64748b; border-radius: 15px;
            }
            #themeButton:hover { background-color: #64748b; }
            #dropArea { 
                background-color: #334155; border: 2px dashed #64748b; border-radius: 4px; 
                color: #94a3b8; font-size: 11px;
            }
            #imageDisplay { 
                background-color: #334155; border: 1px solid #64748b; border-radius: 4px; 
                color: #94a3b8; font-size: 12px;
            }
            #compactStatusBar { background-color: #0f172a; color: #94a3b8; font-size: 10px; }
        """
    
    def apply_theme(self):
        """Apply current theme with appropriate icons"""
        if self.dark_theme:
            self.setStyleSheet(self.get_dark_theme_style())
            self.theme_btn.setIcon(qta.icon('fa5s.sun', color='#cbd5e1'))
        else:
            self.setStyleSheet(self.get_light_theme_style())
            self.theme_btn.setIcon(qta.icon('fa5s.moon', color='#6b7280'))
    
    def toggle_theme(self):
        """Toggle between themes"""
        self.dark_theme = not self.dark_theme
        self.apply_theme()
        self.save_config()
    
    def browse_image(self):
        """Browse for image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.handle_dropped_file(file_path)
    
    def handle_dropped_file(self, file_path):
        """Handle dropped or browsed image file"""
        try:
            self.drop_area.setText(f"📁 {os.path.basename(file_path)}")
            self.status_bar.showMessage("🔍 Analyzing image...")
            
            # Show preview if dropped on image display
            if self.sender() == self.image_label:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap)
            
            # Start image recognition
            self.worker = ImageWorker(self.api_key_input.text().strip(), 'recognize', file_path)
            self.worker.image_recognized.connect(self.on_image_recognized)
            self.worker.error_occurred.connect(self.on_error)
            self.worker.start()
            
        except Exception as e:
            self.show_error(f"Failed to process image: {str(e)}")
    
    def on_image_recognized(self, description):
        """Handle successful image recognition"""
        self.context_output.setText(description)
        self.use_context_btn.setEnabled(True)
        self.status_bar.showMessage("✅ Image analyzed", 2000)
    
    def use_context(self):
        """Use context description as prompt"""
        context = self.context_output.toPlainText()
        current_prompt = self.prompt_input.toPlainText()
        
        if current_prompt.strip():
            combined = f"{current_prompt}\n\nBased on: {context}"
        else:
            combined = context
            
        self.prompt_input.setText(combined)
    
    def validate_inputs(self):
        """Validate inputs"""
        if not self.api_key_input.text().strip() and GEMINI_AVAILABLE:
            self.show_error("Enter API key")
            return False
        
        if not self.prompt_input.toPlainText().strip():
            self.show_error("Enter prompt")
            return False
        
        return True
    
    def generate_image(self):
        """Generate image"""
        if not self.validate_inputs():
            return
        
        self.last_prompt = self.prompt_input.toPlainText().strip()
        self.start_generation()
    
    def regenerate_image(self):
        """Regenerate with last prompt"""
        if not self.last_prompt:
            self.show_error("No previous prompt")
            return
        
        if not self.api_key_input.text().strip() and GEMINI_AVAILABLE:
            self.show_error("Enter API key")
            return
        
        self.start_generation()
    
    def start_generation(self):
        """Start generation process"""
        self.generate_btn.setEnabled(False)
        self.regenerate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        self.status_bar.showMessage("🎨 Generating...")
        
        self.worker = ImageWorker(self.api_key_input.text().strip(), 'generate', self.last_prompt)
        self.worker.image_generated.connect(self.on_image_generated)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def on_image_generated(self, image):
        """Handle generated image"""
        try:
            self.current_image = image
            
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(img_byte_arr.getvalue())
            
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            self.generate_btn.setEnabled(True)
            self.regenerate_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            self.status_bar.showMessage("✅ Generated!", 2000)
            
        except Exception as e:
            self.on_error(f"Display failed: {str(e)}")
    
    def on_error(self, error_message):
        """Handle errors"""
        self.show_error(error_message)
        
        self.generate_btn.setEnabled(True)
        if self.last_prompt:
            self.regenerate_btn.setEnabled(True)
        
        self.status_bar.showMessage("❌ Failed", 2000)
    
    def save_image(self):
        """Save current image"""
        if not self.current_image:
            self.show_error("No image to save")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "generated.png",
                "PNG (*.png);;JPEG (*.jpg)"
            )
            
            if file_path:
                self.current_image.save(file_path)
                self.status_bar.showMessage(f"💾 Saved: {os.path.basename(file_path)}", 2000)
                
        except Exception as e:
            self.show_error(f"Save failed: {str(e)}")
    
    def load_config(self):
        """Load configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.api_key_input.setText(config.get('api_key', ''))
                    self.dark_theme = config.get('dark_theme', True)
        except Exception as e:
            self.show_error(f"Config load failed: {str(e)}")
    
    def save_config(self):
        """Save configuration"""
        try:
            config = {
                'api_key': self.api_key_input.text(),
                'dark_theme': self.dark_theme
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            pass
    
    def show_error(self, message):
        """Show compact error"""
        QMessageBox.critical(self, "Error", message)
    
    def clear_status(self):
        """Clear status"""
        self.status_bar.clearMessage()

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Gemini Image Generator")
    window = GeminiImageGenerator()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()