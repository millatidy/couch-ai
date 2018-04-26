<template>
  <div>
    <p>Chat page</p>
    <input type="text" class="form-control" id="message" placeholder="Enter message">
    <p>Reply from backend: {{ reply }}</p>
    <button @click="getReply">Send</button>
  </div>
</template>

<script>
import axios from 'axios'
export default {
  data () {
    return {
      reply: 'none'
    }
  },
  methods: {
    getReply () {
      // this.reply = this.getFakeReply()
      this.reply = this.getReplyFromBackend()
      console.log('Calling getReply')
    },
    getFakeReply () {
      return 'Hello'
    },
    getReplyFromBackend () {
      console.log('Calling backend function')
      const path = `http://localhost:5000/api/chat`
      var message = document.getElementById('message').value
      console.log(message)
      axios.post(path, {
        message: message
      })
        .then(response => {
          this.reply = response.data.message
        })
        .catch(error => {
          console.log(error)
        })
    }
  },
  created () {
    this.getReply()
  }
}
</script>
