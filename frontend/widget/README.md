# Biometric Text Authentication Widget

A minimal React widget for text-based biometric authentication with keystroke timing analysis.

## Features

- ✅ **Paste Prevention**: Prevents paste events and flags attempts
- ✅ **Keystroke Timing**: Collects inter-key intervals and creates privacy-preserving histograms
- ✅ **Auto-submit Timer**: Visual countdown with auto-submit on timeout
- ✅ **Enrollment Flow**: Guided multi-sample enrollment process
- ✅ **Verification Flow**: Real-time authentication verification
- ✅ **Word Count Tracking**: Real-time word counter with minimum requirements
- ✅ **Session Management**: Token-based session handling with backend

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
cd frontend/widget
npm install
```

### Development

```bash
npm run dev
```

The widget will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The optimized build will be in the `dist/` directory.

## Usage

### Enrollment Mode

```jsx
import BiometricTextWidget from './Widget';

<BiometricTextWidget
  mode="enroll"
  userId="user_123"
  apiBase="/api"
/>
```

**Flow**:
1. Widget calls `/api/enroll/start` to get challenges
2. User types response (8 samples required)
3. Each submission posts to `/api/enroll/submit`
4. After 8 samples, profile is created

### Verification Mode

```jsx
<BiometricTextWidget
  mode="verify"
  userId="user_123"
  apiBase="/api"
/>
```

**Flow**:
1. User types naturally in response to prompt
2. Widget submits to `/api/verify`
3. Decision returned: `allow`, `challenge`, `step_up`, or `deny`

## Keystroke Timing Collection

The widget collects keystroke timing data in a privacy-preserving manner:

### What is Collected
- **Inter-key intervals (IKI)**: Time between consecutive key presses
- **Histogram bins**: [0-50ms, 50-100ms, 100-150ms, 150-200ms, 200-300ms, 300+ms]
- **Statistics**: Mean and standard deviation of IKIs

### What is NOT Collected
- Individual keystroke timestamps
- Specific keys pressed (only for timing)
- Raw timing data

### Privacy Features
- Only binned histograms are sent to server
- No personally identifiable keystroke patterns
- Pauses > 5 seconds excluded from analysis
- Client-side aggregation before transmission

## API Integration

### Backend Endpoints Expected

#### POST `/api/enroll/start`
Request:
```json
{
  "user_id": "string",
  "lang": "en",
  "domain": "chat"
}
```

Response:
```json
{
  "challenges": [{
    "challenge_id": "uuid",
    "prompt": "Write about...",
    "min_words": 70,
    "timebox_s": 180,
    "constraints": ["no_paste", "min_sentences_3"]
  }],
  "session_token": "uuid",
  "required_samples": 8
}
```

#### POST `/api/enroll/submit`
Request:
```json
{
  "user_id": "string",
  "challenge_id": "uuid",
  "text": "user's typed text...",
  "timings": {
    "histogram": [5, 10, 15, 8, 3, 1],
    "mean_iki": 150,
    "std_iki": 45,
    "total_events": 42
  },
  "session_token": "uuid"
}
```

Response:
```json
{
  "accepted": true,
  "remaining": 7,
  "warnings": [],
  "profile_ready": false
}
```

#### POST `/api/verify`
Request:
```json
{
  "user_id": "string",
  "text": "user's typed text...",
  "timings": {...},
  "lang": "en",
  "domain_hint": "chat"
}
```

Response:
```json
{
  "decision": "allow",
  "score": 0.87,
  "reasons": ["HIGH_CONFIDENCE"],
  "thresholds": {"high": 0.84, "med": 0.72}
}
```

## Component Props

### BiometricTextWidget

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `mode` | `'enroll' \| 'verify'` | `'enroll'` | Widget mode |
| `userId` | `string` | required | User identifier |
| `apiBase` | `string` | `'/api'` | API base URL |

## Testing

### Manual Testing
1. Start backend: `uvicorn app:app --reload`
2. Start widget: `npm run dev`
3. Navigate to `http://localhost:3000`
4. Test enrollment flow (8 samples)
5. Test verification flow

### Unit Tests
```bash
npm test
```

Tests cover:
- Keystroke histogram generation
- Paste prevention
- Timer functionality
- Word count validation

## Security Features

### Paste Prevention
- `onPaste` event handler prevents paste
- Client-side tracking of paste attempts
- Visual warning to user

### Privacy Protection
- Only histogram data sent to server
- No raw keystroke timestamps
- No key content recorded
- Timing data binned before transmission

### Timeout Protection
- Auto-submit after time limit
- Visual countdown warning
- Prevents indefinite sessions

## Architecture

```
Widget
├── App.jsx              # Main app with mode selector
├── Widget.jsx           # Core widget component
├── keystroke.js         # Keystroke timing collector
├── Widget.css           # Widget styles
└── App.css              # App layout styles
```

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Known Limitations

1. **Mobile Support**: Keystroke timing less reliable on mobile keyboards
2. **Virtual Keyboards**: May not work with on-screen keyboards
3. **Browser Extensions**: Some extensions may interfere with paste prevention
4. **Autocomplete**: Browser autocomplete may affect timing data

## Future Enhancements

- [ ] Mobile-optimized UI
- [ ] Multi-language support
- [ ] Accessibility improvements (WCAG 2.1)
- [ ] Dark mode support
- [ ] Progressive Web App (PWA) features
- [ ] Offline mode with sync

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: [biometric-text-analysis/issues]
- Email: support@example.com

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.
