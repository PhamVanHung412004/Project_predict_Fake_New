import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  Tooltip,
  Fade
} from '@mui/material';
import {
  Send as SendIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  SmartToy as BotIcon,
  Person as PersonIcon
} from '@mui/icons-material';
import axios from 'axios';
import WelcomeScreen from './WelcomeScreen';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  analysis?: {
    prediction: number;
    label: string;
    confidence: number;
    analysis?: any;
  };
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSendMessage = async () => {
    if (!inputText.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputText.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/predict', {
        text: inputText.trim(),
        include_analysis: true
      });

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: `I have analyzed your text. Result: **${response.data.label}** with confidence **${response.data.confidence}%**`,
        timestamp: new Date(),
        analysis: {
          prediction: response.data.prediction,
          label: response.data.label,
          confidence: response.data.confidence,
          analysis: response.data.analysis
        }
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err: any) {
      setError(err.response?.data?.error || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setError(null);
    inputRef.current?.focus();
  };

  const handlePromptClick = (prompt: string) => {
    setInputText(prompt);
    inputRef.current?.focus();
  };

  const getMessageIcon = (type: 'user' | 'bot') => {
    return type === 'user' ? <PersonIcon /> : <BotIcon />;
  };

  const getMessageColor = (type: 'user' | 'bot') => {
    return type === 'user' ? 'primary' : 'secondary';
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('vi-VN', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      bgcolor: '#212121'
    }}>
      {/* Clear Chat Button */}
      <Box sx={{ 
        p: 2, 
        display: 'flex', 
        justifyContent: 'flex-end',
        borderBottom: '1px solid #3d3d3d'
      }}>
        <Tooltip title="Clear conversation">
          <IconButton onClick={handleClearChat} sx={{ color: 'white' }}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Messages Area */}
      <Box sx={{ 
        flex: 1, 
        overflow: 'auto', 
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        gap: 2
      }}>
        {messages.length === 0 && <WelcomeScreen onPromptClick={handlePromptClick} />}

        {messages.map((message) => (
          <Fade in={true} key={message.id}>
            <Box sx={{ 
              display: 'flex', 
              gap: 2,
              justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
              mb: 2
            }}>
              {message.type === 'bot' && (
                <Avatar sx={{ bgcolor: 'secondary.main' }}>
                  {getMessageIcon(message.type)}
                </Avatar>
              )}
              
              <Box sx={{ 
                maxWidth: '70%',
                display: 'flex',
                flexDirection: 'column',
                gap: 1
              }}>
                <Paper sx={{ 
                  p: 2,
                  bgcolor: message.type === 'user' ? '#2f2f2f' : '#3d3d3d',
                  color: 'white',
                  borderRadius: 3,
                  boxShadow: 0,
                  border: '1px solid',
                  borderColor: message.type === 'user' ? '#404040' : '#4d4d4d'
                }}>
                  <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {message.content}
                  </Typography>
                </Paper>

                {/* Analysis Results */}
                {message.analysis && (
                  <Box sx={{ mt: 1 }}>
                    <Chip
                      icon={message.analysis.prediction === 0 ? <WarningIcon /> : <CheckCircleIcon />}
                      label={`${message.analysis.label} (${message.analysis.confidence}%)`}
                      color={message.analysis.prediction === 0 ? 'error' : 'success'}
                      variant="outlined"
                      sx={{ mb: 1 }}
                    />
                    
                    {message.analysis.analysis?.success && message.analysis.analysis.analysis && (
                      <Paper sx={{ 
                        p: 2, 
                        bgcolor: '#2a2a2a', 
                        borderRadius: 2,
                        border: '1px solid #404040'
                      }}>
                        <Typography variant="subtitle2" gutterBottom sx={{ color: 'white' }}>
                          📊 Detailed Analysis:
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#b0b0b0' }}>
                          {message.analysis.analysis.analysis.summary}
                        </Typography>
                      </Paper>
                    )}
                  </Box>
                )}

                <Typography variant="caption" sx={{ alignSelf: 'flex-end', color: '#808080' }}>
                  {formatTime(message.timestamp)}
                </Typography>
              </Box>

              {message.type === 'user' && (
                <Avatar sx={{ bgcolor: 'primary.main' }}>
                  {getMessageIcon(message.type)}
                </Avatar>
              )}
            </Box>
          </Fade>
        ))}

        {loading && (
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Avatar sx={{ bgcolor: 'secondary.main' }}>
              <BotIcon />
            </Avatar>
            <Paper sx={{ 
              p: 2, 
              bgcolor: '#3d3d3d', 
              borderRadius: 3,
              border: '1px solid #4d4d4d'
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CircularProgress size={16} sx={{ color: 'white' }} />
                <Typography variant="body2" sx={{ color: 'white' }}>
                  AI is analyzing...
                </Typography>
              </Box>
            </Paper>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 1 }}>
            {error}
          </Alert>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Paper elevation={0} sx={{ 
        p: 2, 
        borderRadius: 0,
        borderTop: '1px solid',
        borderColor: '#3d3d3d',
        bgcolor: '#2f2f2f'
      }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            ref={inputRef}
            fullWidth
            multiline
            maxRows={4}
            variant="outlined"
            placeholder="Enter news to verify..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 3,
                bgcolor: '#3d3d3d',
                color: 'white',
                '& fieldset': {
                  borderColor: '#4d4d4d',
                },
                '&:hover fieldset': {
                  borderColor: '#5d5d5d',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#1976d2',
                }
              },
              '& .MuiInputBase-input': {
                color: 'white',
              },
              '& .MuiInputBase-input::placeholder': {
                color: '#808080',
                opacity: 1
              }
            }}
          />
          <IconButton
            onClick={handleSendMessage}
            disabled={!inputText.trim() || loading}
            sx={{ 
              bgcolor: inputText.trim() && !loading ? '#1976d2' : '#404040',
              color: 'white',
              width: 48,
              height: 48,
              '&:hover': {
                bgcolor: inputText.trim() && !loading ? '#1565c0' : '#404040',
              },
              '&:disabled': {
                bgcolor: '#404040',
                color: '#666'
              },
              transition: 'all 0.2s'
            }}
          >
            <SendIcon />
          </IconButton>
        </Box>
        
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          mt: 1 
        }}>
          <Typography variant="caption" sx={{ color: '#808080' }}>
            Press Enter to send, Shift+Enter for new line
          </Typography>
          <Typography variant="caption" sx={{ color: '#808080' }}>
            {messages.length} messages
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default ChatInterface;
