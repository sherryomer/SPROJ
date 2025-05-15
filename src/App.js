import React, { useState } from 'react';
import axios from 'axios';
import './styles/App.css';
import lumsLogo from './logo/LumsLogo.png';

const App = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [chatHistories, setChatHistories] = useState([]);
    const [activeChatIndex, setActiveChatIndex] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async () => {
        if (input.trim() === '' || isLoading) return;

        const newMessage = { sender: 'user', text: input };
        const updatedMessages = [...messages, newMessage];
        setMessages(updatedMessages);
        setInput('');
        setIsLoading(true);

        try {
            const response = await axios.post('http://127.0.0.1:5000/query', { query: input });
            const botMessage = response.data.answer;
            simulateTypingEffect(botMessage, updatedMessages, input);
        } catch (error) {
            const errorMessage = 'Error: Unable to get a response!';
            simulateTypingEffect(errorMessage, updatedMessages, input);
        } finally {
            setIsLoading(false);
        }
    };

    const simulateTypingEffect = (botMessage, previousMessages, userPrompt) => {
        let currentText = '';
        const typingSpeed = 30;
        const botResponse = { sender: 'bot', text: '' };

        const interval = setInterval(() => {
            currentText += botMessage[currentText.length];
            botResponse.text = currentText;

            setMessages([...previousMessages, botResponse]);

            if (currentText.length === botMessage.length) {
                clearInterval(interval);
                updateChatHistories(userPrompt, [...previousMessages, botResponse]);
            }
        }, typingSpeed);
    };

    const updateChatHistories = (userPrompt, fullChatMessages) => {
        const chatTitle = generateChatTitle(userPrompt);

        if (activeChatIndex === null) {
            const newChat = {
                title: chatTitle || 'New Chat',
                messages: fullChatMessages,
            };
            setChatHistories([...chatHistories, newChat]);
            setActiveChatIndex(chatHistories.length);
        } else {
            const updatedHistories = [...chatHistories];
            updatedHistories[activeChatIndex] = {
                ...updatedHistories[activeChatIndex],
                messages: fullChatMessages,
            };
            setChatHistories(updatedHistories);
        }
    };

    const generateChatTitle = (prompt) => {
        const stopWords = ['tell', 'me', 'about', 'the', 'of', 'and', 'a', 'is'];
        const words = prompt
            .split(' ')
            .filter((word) => !stopWords.includes(word.toLowerCase()))
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
        return words.join(' ').slice(0, 30);
    };

    const startNewChat = () => {
        setMessages([]);
        setActiveChatIndex(null);
    };

    const selectChat = (index) => {
        setActiveChatIndex(index);
        setMessages(chatHistories[index].messages);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !isLoading) {
            sendMessage();
        }
    };

    return (
        <div className="chat-app">
            <div className="sidebar">
                <h2 className="sidebar-title">LUMS Assistant</h2>
                <button onClick={startNewChat} className="new-chat-btn">
                    <span className="plus-icon">+</span> New Chat
                </button>
                <div className="chat-history">
                    <h3>Recent Chats</h3>
                    <ul>
                        {chatHistories.map((chat, index) => (
                            <li
                                key={index}
                                className={activeChatIndex === index ? 'active' : ''}
                                onClick={() => selectChat(index)}
                            >
                                <span className="chat-icon">💬</span>
                                {chat.title}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            <div className="chat-container">
                <div className="chat-header">
                    <img src={lumsLogo} alt="LUMS Logo" className="lums-logo" />
                </div>

                <div className="chat-window">
                    {messages.length === 0 ? (
                        <div className="welcome-message">
                            <h3>Hello! I'm LUMS Assistant</h3>
                            <p>Ask me anything about LUMS, including:</p>
                            <ul>
                                <li>Admission requirements</li>
                                <li>Campus facilities</li>
                                <li>Academic programs</li>
                                <li>Student services</li>
                            </ul>
                        </div>
                    ) : (
                        messages.map((msg, index) => (
                            <div
                                key={index}
                                className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
                            >
                                <div className="message-content">
                                    {msg.text}
                                </div>
                            </div>
                        ))
                    )}
                </div>

                <div className="input-container">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about LUMS..."
                        disabled={isLoading}
                    />
                    <button 
                        onClick={sendMessage} 
                        className={isLoading ? 'loading' : ''}
                        disabled={isLoading}
                    >
                        {isLoading ? '...' : 'Send'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default App;
