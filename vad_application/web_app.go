package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/improbable-eng/grpc-web/go/grpcweb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/your-org/vad-service/vad"
)

// VADServer implements the VADServiceServer
type VADServer struct {
	pb.UnimplementedVADServiceServer
}

// ProcessAudio handles streaming audio and detects voice activity
func (s *VADServer) ProcessAudio(stream pb.VADService_ProcessAudioServer) error {
	var isSpeaking bool
	var silenceStart time.Time
	const silenceThreshold = 2 * time.Second
	const amplitudeThreshold = 1000 // Simple amplitude threshold for demo

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			return stream.SendAndClose(&pb.VADResponse{
				Event:   "end",
				Message: "Stream ended",
			})
		}
		if err != nil {
			return status.Errorf(codes.Internal, "failed to receive audio chunk: %v", err)
		}

		// Simple VAD logic: analyze audio amplitude
		// In a real implementation, use Silero VAD or similar
		amplitude := calculateAmplitude(chunk.AudioData)
		event := ""
		message := ""

		if amplitude > amplitudeThreshold {
			if !isSpeaking {
				isSpeaking = true
				event = "start"
				message = "Speech detected"
			} else {
				event = "continue"
				message = "Speech continuing"
			}
			silenceStart = time.Time{}
		} else {
			if isSpeaking {
				if silenceStart.IsZero() {
					silenceStart = time.Now()
				} else if time.Since(silenceStart) > silenceThreshold {
					isSpeaking = false
					event = "end"
					message = "Speech ended"
				}
			}
		}

		if event != "" {
			if err := stream.Send(&pb.VADResponse{Event: event, Message: message}); err != nil {
				return status.Errorf(codes.Internal, "failed to send response: %v", err)
			}
		}
	}
}

// ResetVAD resets the VAD state
func (s *VADServer) ResetVAD(ctx context.Context, req *pb.ResetRequest) (*pb.ResetResponse, error) {
	return &pb.ResetResponse{Success: true}, nil
}

// calculateAmplitude is a simple demo function to analyze audio data
func calculateAmplitude(data []byte) int32 {
	if len(data) == 0 {
		return 0
	}
	var sum int32
	for i := 0; i < len(data); i += 2 {
		if i+1 < len(data) {
			// Interpret as 16-bit samples
			sample := int32(data[i]) | (int32(data[i+1]) << 8)
			if sample > 32767 {
				sample -= 65536
			}
			if sample < 0 {
				sample = -sample
			}
			sum += sample
		}
	}
	return sum / int32(len(data)/2)
}

func main() {
	// Set up gRPC server
	grpcServer := grpc.NewServer()
	pb.RegisterVADServiceServer(grpcServer, &VADServer{})

	// Wrap gRPC server with gRPC-Web
	grpcWebServer := grpcweb.WrapServer(grpcServer,
		grpcweb.WithOriginFunc(func(origin string) bool { return true }),
	)

	// Set up HTTP server with Gorilla Mux
	r := mux.NewRouter()

	// Serve static files (HTML, JS)
	r.PathPrefix("/static/").Handler(http.StripPrefix("/static/", http.FileServer(http.Dir("./static"))))

	// Handle gRPC-Web requests
	r.HandleFunc("/vad.VADService/{method}", func(w http.ResponseWriter, r *http.Request) {
		if grpcWebServer.IsGrpcWebRequest(r) {
			grpcWebServer.ServeHTTP(w, r)
		} else {
			http.Error(w, "Not a gRPC-Web request", http.StatusBadRequest)
		}
	})

	// Root route
	r.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "./static/index.html")
	})

	// Start server
	fmt.Println("Server starting on :8080...")
	if err := http.ListenAndServe(":8080", r); err != nil {
		log.Fatalf("failed to start server: %v", err)
	}
}
