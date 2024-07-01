package main



import (
	//"encoding/json"
	"fmt"
	"log"
	"net/http"
	"github.com/gorilla/mux"
	"github.com/spf13/viper"

)

func helloTest(w  http.ResponseWriter, r * http.Request){

	fmt.Println("Hello World")

}
func main(){
	viper.SetConfigType("yaml")
	viper.SetConfigName("config")
	viper.AddConfigPath(".")
	err := viper.ReadInConfig()
	if err != nil{
		log.Fatal(fmt.Errorf("fatal error config file:%w",err))
	}
	//create a new router
	router := mux.NewRouter();

	//Define Handler function for API endpoints
	router.HandleFunc("/hello",helloTest).Methods(http.MethodGet)

	//running server
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d",viper.GetInt("port")),router))
	
}

