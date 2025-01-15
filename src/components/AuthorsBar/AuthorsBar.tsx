import React from 'react';

const AuthorsBar: React.FC<AuthorsBarProps> = ({ authors }) => {
    return (
      <div  style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '2rem',
        flexWrap: 'wrap',
      }}>
        {authors.map((author, index) => (
          <div
            key={index}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <a
              href={author.link}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                textDecoration: 'none',
                color: '#000',
                fontWeight: 'bold',
              }}
            >
              {author.lastName}, {author.firstName}
            </a>
            <span style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>
              {author.university}
            </span>
          </div>
        ))}
      </div>
    );
  };
  
  export default AuthorsBar;