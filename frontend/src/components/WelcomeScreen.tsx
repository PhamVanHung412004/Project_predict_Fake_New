import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Avatar
} from '@mui/material';
import {
  SmartToy as BotIcon,
  Article as ArticleIcon,
  Speed as SpeedIcon,
  Security as SecurityIcon
} from '@mui/icons-material';

interface WelcomeScreenProps {
  onPromptClick?: (prompt: string) => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onPromptClick }) => {
  const features = [
    {
      icon: <ArticleIcon />,
      title: "Phân tích văn bản",
      description: "Kiểm tra độ tin cậy của tin tức"
    },
    {
      icon: <SpeedIcon />,
      title: "Tốc độ nhanh",
      description: "Kết quả trong vài giây"
    },
    {
      icon: <SecurityIcon />,
      title: "Bảo mật cao",
      description: "Dữ liệu được bảo vệ an toàn"
    }
  ];

  const examplePrompts = [
    "Kiểm tra tin tức này có đáng tin không?",
    "Phân tích độ tin cậy của bài báo này",
    "Đây có phải là tin giả không?",
    "Hãy kiểm tra thông tin này"
  ];

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      height: '100%',
      textAlign: 'center',
      gap: 4,
      p: 3
    }}>
      {/* Main Welcome */}
      <Box>
        <Avatar sx={{ 
          width: 80, 
          height: 80, 
          bgcolor: 'primary.main',
          mx: 'auto',
          mb: 2
        }}>
          <BotIcon sx={{ fontSize: 40 }} />
        </Avatar>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 600, color: 'white' }}>
          Chào mừng đến với AI Phát Hiện Tin Giả
        </Typography>
        <Typography variant="h6" sx={{ mb: 3, color: '#b0b0b0' }}>
          Tôi có thể giúp bạn kiểm tra độ tin cậy của bất kỳ tin tức nào
        </Typography>
      </Box>

      {/* Features */}
      <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', sm: 'row' },
        gap: 2, 
        maxWidth: 600,
        width: '100%'
      }}>
        {features.map((feature, index) => (
          <Box key={index} sx={{ flex: 1 }}>
            <Card sx={{ 
              height: '100%', 
              textAlign: 'center',
              bgcolor: '#2f2f2f',
              border: '1px solid #404040'
            }}>
              <CardContent>
                <Avatar sx={{ 
                  bgcolor: 'primary.main', 
                  mx: 'auto', 
                  mb: 1,
                  width: 48,
                  height: 48
                }}>
                  {feature.icon}
                </Avatar>
                <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                  {feature.title}
                </Typography>
                <Typography variant="body2" sx={{ color: '#b0b0b0' }}>
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Box>
        ))}
      </Box>

      {/* Example Prompts */}
      <Box>
        <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
          Bạn có thể hỏi tôi:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, justifyContent: 'center' }}>
          {examplePrompts.map((prompt, index) => (
            <Chip
              key={index}
              label={prompt}
              variant="outlined"
              onClick={() => onPromptClick?.(prompt)}
              sx={{ 
                cursor: 'pointer',
                borderColor: '#4d4d4d',
                color: '#b0b0b0',
                '&:hover': {
                  bgcolor: '#3d3d3d',
                  borderColor: '#1976d2',
                  color: 'white'
                }
              }}
            />
          ))}
        </Box>
      </Box>

      {/* Instructions */}
      <Box sx={{ 
        bgcolor: '#2a2a2a', 
        p: 3, 
        borderRadius: 2,
        maxWidth: 500,
        width: '100%',
        border: '1px solid #404040'
      }}>
        <Typography variant="body1" sx={{ color: '#b0b0b0' }}>
          💡 <strong style={{ color: 'white' }}>Mẹo sử dụng:</strong> Gửi tin tức hoặc văn bản cho tôi, 
          tôi sẽ phân tích và cho bạn biết độ tin cậy của thông tin đó.
        </Typography>
      </Box>
    </Box>
  );
};

export default WelcomeScreen;
