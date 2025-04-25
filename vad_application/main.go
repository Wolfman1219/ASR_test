// main.go
package main

import (
	"context"
	"log"
	"net/http"

	pb "vad-application/grpc_modules" // replace with your actual path

	"github.com/gorilla/websocket"
	"google.golang.org/grpc"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	defer ws.Close()

	// gRPC client
	conn, err := grpc.NewClient("localhost:50055", grpc.WithInsecure())
	if err != nil {
		log.Fatal("gRPC dial error:", err)
	}
	defer conn.Close()

	client := pb.NewVADServiceClient(conn)
	stream, err := client.ProcessAudio(context.Background())
	if err != nil {
		log.Fatal("gRPC stream error:", err)
	}

	// Send audio from WebSocket to gRPC
	go func() {
		for {
			_, audio, err := ws.ReadMessage()
			if err != nil {
				log.Println("WS read error:", err)
				break
			}
			stream.Send(&pb.AudioChunk{AudioData: audio})
		}
	}()

	// Send VAD response back to browser
	for {
		resp, err := stream.Recv()
		if err != nil {
			log.Println("gRPC recv error:", err)
			break
		}
		ws.WriteJSON(resp)
	}
}

func main() {
	http.Handle("/", http.FileServer(http.Dir("./static")))
	http.HandleFunc("/ws", wsHandler)
	log.Println("Server listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
