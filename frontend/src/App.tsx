import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Typography, Box, Paper } from '@mui/material';
import ChatInterface from './components/ChatInterface';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#212121',
      paper: '#2f2f2f',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 600,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Paper elevation={0} sx={{ 
          borderRadius: 0,
          borderBottom: '1px solid #3d3d3d',
          bgcolor: '#2f2f2f'
        }}>
          <Box sx={{ py: 2, px: 3 }}>
            <Typography 
              variant="h4" 
              component="h1" 
              align="center" 
              gutterBottom
              sx={{ 
                background: 'linear-gradient(45deg, #1976d2, #dc004e)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: 600
              }}
            >
              AI Phát Hiện Tin Giả
            </Typography>
            <Typography 
              variant="subtitle1" 
              align="center" 
              sx={{ 
                mb: 2,
                color: '#b0b0b0'
              }}
            >
              Chat với AI để kiểm tra độ tin cậy của tin tức
            </Typography>
          </Box>
        </Paper>

        {/* Chat Interface */}
        <Box sx={{ flex: 1, overflow: 'hidden' }}>
          <ChatInterface />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;