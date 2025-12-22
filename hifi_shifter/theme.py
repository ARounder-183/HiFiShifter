from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import Qt
import pyqtgraph as pg

# Theme Definitions
THEMES = {
    'dark': {
        'name': 'Dark',
        'palette': {
            QPalette.ColorRole.Window: '#353535',
            QPalette.ColorRole.WindowText: '#CCCCCC',
            QPalette.ColorRole.Base: '#303030',
            QPalette.ColorRole.AlternateBase: '#353535',
            QPalette.ColorRole.ToolTipBase: '#CCCCCC',
            QPalette.ColorRole.ToolTipText: '#CCCCCC',
            QPalette.ColorRole.Text: '#CCCCCC',
            QPalette.ColorRole.Button: '#353535',
            QPalette.ColorRole.ButtonText: '#CCCCCC',
            QPalette.ColorRole.BrightText: '#FF6666',
            QPalette.ColorRole.Link: '#4DA6FF',
            QPalette.ColorRole.Highlight: '#2A82DA',
            QPalette.ColorRole.HighlightedText: '#FFFFFF',
        },
        'graph': {
            'background': '#1e1e1e',
            'foreground': '#AAAAAA',
            'waveform_pen': (255, 255, 255, 100),
            'waveform_brush': (255, 255, 255, 30),
            'f0_orig_pen': (255, 255, 255, 80),
            'f0_pen': '#00ff00',
            'f0_selected_pen': '#0099ff',
            'grid_alpha': 0.2,
        },
        'piano_roll': {
            'grid_bar': (180, 180, 180, 40),
            'grid_beat': (100, 100, 100, 25),
            'grid_sub': (70, 70, 70, 15),
            'black_key': (30, 30, 30, 100),
            'white_key_pen': (60, 60, 60, 30),
            'cursor': '#0099ff',
            'selection_brush': (255, 255, 255, 50),
            'selection_pen': (255, 255, 255, 200),
        },
        'track_control': {
            'bg_selected': '#2D2D2D',
            'bg_normal': '#202020',
            'border_selected': '#666',
            'border_normal': '#444',
            'text': '#cccccc',
            'panel_background': '#202020',
            'buttons': {
                'delete': {'normal': '#888', 'hover': '#f44'},
                'mute': {'bg': '#444', 'checked_bg': '#d44', 'checked_text': 'white'},
                'solo': {'bg': '#444', 'checked_bg': '#dd4', 'checked_text': 'black'},
                'bgm': {'bg': '#444', 'checked_bg': '#4a4', 'checked_text': 'black'}
            },
            'slider': {
                'groove': '#444',
                'handle': '#888',
                'sub_page': '#666'
            }
        },
        'audio_block': {
            'vocal': {
                'bg': (40, 60, 100, 200),
                'wave': (100, 180, 255, 255),
                'border': (100, 150, 255),
            },
            'bgm': {
                'bg': (40, 80, 40, 200),
                'wave': (150, 255, 100, 255),
                'border': (150, 255, 100),
            },
            'selected_border': (255, 255, 255, 150),
        },
        'menu': {
            'background': '#353535',
            'text': '#CCCCCC',
            'selected_background': '#2A82DA',
            'selected_text': '#FFFFFF',
            'border': '#444444'
        }
    },
    'light': {
        'name': 'Light',
        'palette': {
            QPalette.ColorRole.Window: '#F0F0F0',
            QPalette.ColorRole.WindowText: '#333333',
            QPalette.ColorRole.Base: '#FFFFFF',
            QPalette.ColorRole.AlternateBase: '#F7F7F7',
            QPalette.ColorRole.ToolTipBase: '#333333',
            QPalette.ColorRole.ToolTipText: '#333333',
            QPalette.ColorRole.Text: '#333333',
            QPalette.ColorRole.Button: '#E0E0E0',
            QPalette.ColorRole.ButtonText: '#333333',
            QPalette.ColorRole.BrightText: '#CC0000',
            QPalette.ColorRole.Link: '#0066CC',
            QPalette.ColorRole.Highlight: '#0078D7',
            QPalette.ColorRole.HighlightedText: '#FFFFFF',
        },
        'graph': {
            'background': '#FFFFFF',
            'foreground': '#555555',
            'waveform_pen': (0, 0, 0, 100),
            'waveform_brush': (0, 0, 0, 30),
            'f0_orig_pen': (0, 0, 0, 80),
            'f0_pen': '#2E8B57',
            'f0_selected_pen': '#1E90FF',
            'grid_alpha': 0.5,
        },
        'piano_roll': {
            'grid_bar': (50, 50, 50, 80),
            'grid_beat': (100, 100, 100, 50),
            'grid_sub': (180, 180, 180, 50),
            'black_key': (220, 220, 220, 100),
            'white_key_pen': (200, 200, 200, 50),
            'cursor': '#0099ff',
            'selection_brush': (0, 120, 215, 80),
            'selection_pen': (0, 120, 215, 255),
        },
        'track_control': {
            'bg_selected': '#ffffff',
            'bg_normal': '#f5f5f5',
            'border_selected': '#999',
            'border_normal': '#ccc',
            'text': '#333333',
            'panel_background': '#f5f5f5',
            'buttons': {
                'delete': {'normal': '#666', 'hover': '#d00'},
                'mute': {'bg': '#ddd', 'checked_bg': '#d44', 'checked_text': 'white'},
                'solo': {'bg': '#ddd', 'checked_bg': '#d4c455', 'checked_text': 'black'},
                'bgm': {'bg': '#ddd', 'checked_bg': '#8d8', 'checked_text': 'black'}
            },
            'slider': {
                'groove': '#ddd',
                'handle': '#888',
                'sub_page': '#aaa'
            }
        },
        'audio_block': {
            'vocal': {
                'bg': (180, 200, 240, 200),
                'wave': (0, 100, 200, 255),
                'border': (0, 100, 200),
            },
            'bgm': {
                'bg': (180, 230, 180, 200),
                'wave': (0, 150, 0, 255),
                'border': (0, 150, 0),
            },
            'selected_border': (180, 180, 180, 255),
        },
        'menu': {
            'background': '#F0F0F0',
            'text': '#333333',
            'selected_background': '#0078D7',
            'selected_text': '#FFFFFF',
            'border': '#CCCCCC'
        }
    }
}

current_theme_name = 'dark'

def get_current_theme():
    return THEMES.get(current_theme_name, THEMES['dark'])

def apply_theme(app, theme_name='dark'):
    global current_theme_name
    if theme_name not in THEMES:
        theme_name = 'dark'
    
    current_theme_name = theme_name
    theme = THEMES[theme_name]
    
    # Apply Palette
    palette = QPalette()
    for role, color in theme['palette'].items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    
    # Apply PyQtGraph Config
    pg.setConfigOption('background', theme['graph']['background'])
    pg.setConfigOption('foreground', theme['graph']['foreground'])
    
    # Apply Menu Stylesheet
    menu_style = f"""
        QMenu {{
            background-color: {theme['menu']['background']};
            color: {theme['menu']['text']};
            border: 1px solid {theme['menu']['border']};
        }}
        QMenu::item {{
            background-color: transparent;
            padding: 4px 20px;
            margin: 2px 4px;
        }}
        QMenu::item:selected {{
            background-color: {theme['menu']['selected_background']};
            color: {theme['menu']['selected_text']};
        }}
        QMenuBar {{
            background-color: {theme['menu']['background']};
            color: {theme['menu']['text']};
        }}
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 10px;
        }}
        QMenuBar::item:selected {{
            background-color: {theme['menu']['selected_background']};
            color: {theme['menu']['selected_text']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {theme['menu']['background']};
            color: {theme['menu']['text']};
            selection-background-color: {theme['menu']['selected_background']};
            selection-color: {theme['menu']['selected_text']};
            border: 1px solid {theme['menu']['border']};
            outline: none;
        }}
        QSpinBox, QDoubleSpinBox {{
            background-color: {theme['palette'][QPalette.ColorRole.Base]};
            color: {theme['palette'][QPalette.ColorRole.Text]};
            border: 1px solid {theme['menu']['border']};
            border-radius: 4px;
            padding: 2px;
        }}
    """
    app.setStyleSheet(menu_style)
    
    return theme
