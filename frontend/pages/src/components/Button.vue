<template>
    <div>
        <div v-if="is_file" class="btn unSelect"
             :style="{color:color, background:background, borderColor:border_color,fontSize:font_size, width:width}"
             @click="open_file">
            <input type="file" v-bind:accept="file_accept" @change="clicked($event)" id="file">
            {{value}}
        </div>
        <div v-else class="btn unSelect" @click="clicked($event)"
             :style="{color:color, background:background, borderColor:border_color,fontSize:font_size, width:width}">
            {{value}}
        </div>
    </div>
</template>

<script>
    export default {
        name: "Button",
        props: {
            value: {
                type: String,
                required: true,
                default: ''
            },
            clicked: {
                type: Function,
                default: () => {
                },
            },
            color: {
                type: String,
                default: 'black'
            },
            background: {
                type: String,
                default: 'white'
            },
            border_color: {
                type: String,
                default: 'black'
            },
            font_size: {
                type: String,
                default: "1.5rem"
            },
            width: {
                type: String,
                default: ''
            },
            is_file: {
                type: Boolean,
                default: false
            },
            file_accept: {
                type: String,
                default: '*'
            }
        },
        methods: {
            open_file: function () {
                let f = document.getElementById("file");
                f.click();
            },
        }
    }
</script>

<style scoped>
    .btn {
        display: inline-block;
        padding: 5px 30px;

        box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
        border-width: 1px;
        border-style: solid;
        border-radius: 10px;

        cursor: pointer;
        text-align: center;
    }

    .btn input {
        display: none
    }

    .btn:hover {
        box-shadow: 0 2px 12px 2px rgba(0, 0, 0, 0.3);
    }

    .btn:active {
        box-shadow: 0 0 0 0 rgba(0, 0, 0, 0);
    }
</style>