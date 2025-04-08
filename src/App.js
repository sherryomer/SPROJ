import React, { useState } from 'react';
import axios from 'axios';
import './styles/App.css';

const App = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [chatHistories, setChatHistories] = useState([]);
    const [activeChatIndex, setActiveChatIndex] = useState(null);

    const sendMessage = async () => {
        if (input.trim() === '') return;

        const newMessage = { sender: 'user', text: input };

        // Add user message to messages
        const updatedMessages = [...messages, newMessage];
        setMessages(updatedMessages);

        setInput('');

        try {
            const response = await axios.post('http://127.0.0.1:5000/query', { query: input });

            const botMessage = response.data.answer;
            simulateTypingEffect(botMessage, updatedMessages, input);
        } catch (error) {
            const errorMessage = 'Error: Unable to get a response!';
            simulateTypingEffect(errorMessage, updatedMessages, input);
        }
    };

    const simulateTypingEffect = (botMessage, previousMessages, userPrompt) => {
        let currentText = '';
        const typingSpeed = 30; // Adjust typing speed here (ms per character)
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
        const chatTitle = generateChatTitle(userPrompt); // Generate a meaningful title

        if (activeChatIndex === null) {
            const newChat = {
                title: chatTitle || 'New Chat',
                messages: fullChatMessages,
            };
            setChatHistories([...chatHistories, newChat]);
            setActiveChatIndex(chatHistories.length); // Make the new chat active
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
        // Simple logic to create a meaningful title
        const stopWords = ['tell', 'me', 'about', 'the', 'of', 'and', 'a', 'is']; // Common stop words to exclude
        const words = prompt
            .split(' ')
            .filter((word) => !stopWords.includes(word.toLowerCase())) // Filter out stop words
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()); // Capitalize words
        return words.join(' ').slice(0, 30); // Join words and truncate to 30 characters
    };

    const startNewChat = () => {
        setMessages([]);
        setActiveChatIndex(null); // Start a new chat
    };

    const selectChat = (index) => {
        setActiveChatIndex(index);
        setMessages(chatHistories[index].messages);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <div className="chat-app">
            <div className="sidebar">
                <h2>Chat Histories</h2>
                <button onClick={startNewChat}>+ New Chat</button>
                <ul>
                    {chatHistories.map((chat, index) => (
                        <li
                            key={index}
                            className={activeChatIndex === index ? 'active' : ''}
                            onClick={() => selectChat(index)}
                        >
                            {chat.title}
                        </li>
                    ))}
                </ul>
            </div>
            <div className="chat-container">
                <div className="chat-window">
                    {messages.map((msg, index) => (
                        <div
                            key={index}
                            className={msg.sender === 'user' ? 'user-message' : 'bot-message'}
                        >
                            {msg.text}
                        </div>
                    ))}
                </div>
                <div className="input-container">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown} // Added this for Enter key functionality
                        placeholder="Type your message..."
                    />
                    <button onClick={sendMessage}>Send</button>
                </div>
            </div>
        </div>
    );
};

export default App;
